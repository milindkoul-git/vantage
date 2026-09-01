/**
 * Load the dashboard in a real browser and fail on anything it complains about.
 *
 * This exists because the whole Python suite once passed against a completely
 * dead page. A temporal dead zone error at module scope threw before the first
 * render; the server was fine, the JSON was fine, `node --check` was fine, and
 * the browser showed a blank rectangle. Nothing that inspects HTML or JSON can
 * see that. Only running it can.
 *
 * Usage, with a dashboard already serving:
 *
 *     node scripts/smoke.mjs [--url http://localhost:8080] [--shots out/dir]
 *
 * Exits non-zero on any page error, any console error, or any workspace that
 * fails to render its own root element. Screenshots are written only when
 * --shots is given; the check does not depend on anyone looking at them.
 */

import { existsSync, mkdirSync } from 'node:fs';
import { argv, exit } from 'node:process';
import puppeteer from 'puppeteer-core';

const arg = (name, fallback) => {
  const index = argv.indexOf(`--${name}`);
  return index >= 0 && argv[index + 1] ? argv[index + 1] : fallback;
};

const URL = arg('url', 'http://localhost:8080');
const SHOTS = arg('shots', null);
const WORKSPACES = ['live', 'incidents', 'trends', 'intelligence', 'investigate', 'twin'];

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  '/usr/bin/google-chrome',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
].filter(Boolean);

const chrome = CHROME_CANDIDATES.find((path) => existsSync(path));
if (!chrome) {
  console.error('No Chrome found. Set CHROME_PATH to its executable.');
  exit(2);
}

const problems = [];

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: 'new',
  args: ['--no-sandbox', '--disable-gpu'],
});

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 900 });

  page.on('pageerror', (error) => problems.push(`pageerror: ${error.message}`));
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    const text = message.text();
    // A failed /api/* fetch is the dashboard correctly reporting that a
    // subsystem is not attached; the page renders an explanation for it. Only
    // errors from the page's own code are failures here.
    if (/Failed to load resource/i.test(text)) return;
    problems.push(`console.error: ${text}`);
  });

  if (SHOTS) mkdirSync(SHOTS, { recursive: true });

  for (const workspace of WORKSPACES) {
    await page.goto(`${URL}/#${workspace}`, { waitUntil: 'networkidle2', timeout: 30_000 });
    // React needs a tick past network idle to paint the resolved state.
    await new Promise((resolve) => setTimeout(resolve, 1200));

    const rendered = await page.evaluate(() => {
      const root = document.getElementById('root');
      return {
        children: root ? root.childElementCount : 0,
        text: (document.body.innerText || '').slice(0, 400),
        crashed: (document.body.innerText || '').includes('Something went wrong'),
      };
    });

    if (rendered.children === 0) problems.push(`${workspace}: #root is empty`);
    if (rendered.crashed) problems.push(`${workspace}: error boundary caught a render failure`);

    if (SHOTS) await page.screenshot({ path: `${SHOTS}/${workspace}.png` });
    console.log(`${workspace.padEnd(13)} ${rendered.children} root children`);
  }

  // The analytics controls are the ones that were previously satisfied by a
  // hidden decoy block; assert they are on screen and operable.
  await page.goto(`${URL}/#trends`, { waitUntil: 'networkidle2' });
  await new Promise((resolve) => setTimeout(resolve, 800));
  const controls = await page.evaluate(() => {
    const check = (id) => {
      const element = document.getElementById(id);
      if (!element) return 'missing';
      const style = window.getComputedStyle(element);
      if (style.display === 'none' || style.visibility === 'hidden') return 'hidden';
      return element.tagName.toLowerCase();
    };
    return { metric: check('an-metric'), since: check('an-since') };
  });
  for (const [name, state] of Object.entries(controls)) {
    if (state !== 'select') problems.push(`analytics control ${name} is ${state}`);
  }
  console.log(`controls      metric=${controls.metric} since=${controls.since}`);

  // The skip link is the first thing a keyboard reaches and is invisible until
  // it has focus. Both halves matter: one that never shows is useless, one that
  // always shows is clutter.
  await page.goto(`${URL}/#live`, { waitUntil: 'networkidle2' });
  const before = await page.evaluate(() => {
    const link = document.querySelector('.skip-link');
    if (!link) return null;
    link.focus();
    return {
      hiddenUntilFocused: link.getBoundingClientRect().top < 0,
      target: link.getAttribute('href'),
      targetExists: Boolean(document.querySelector(link.getAttribute('href'))),
    };
  });
  // The link slides in over 120ms, so it has to be measured after that rather
  // than in the same tick as the focus() that started it.
  await new Promise((resolve) => setTimeout(resolve, 300));
  const skip = before === null
    ? { present: false }
    : {
        present: true,
        ...before,
        visibleWhenFocused: await page.evaluate(
          () => document.querySelector('.skip-link').getBoundingClientRect().top >= 0,
        ),
      };
  if (!skip.present) problems.push('no skip link');
  else {
    if (!skip.hiddenUntilFocused) problems.push('skip link is visible before focus');
    if (!skip.visibleWhenFocused) problems.push('skip link stays hidden when focused');
    if (!skip.targetExists) problems.push(`skip link points at ${skip.target}, which is not there`);
    console.log(`skip link     ${skip.target} -> ${skip.targetExists ? 'ok' : 'MISSING'}`);
  }

  // Reduced motion is a request to remove animation, not to hurry it. Anything
  // still running a transition longer than a frame has ignored it.
  await page.emulateMediaFeatures([
    { name: 'prefers-reduced-motion', value: 'reduce' },
  ]);
  await page.goto(`${URL}/#trends`, { waitUntil: 'networkidle2' });
  await new Promise((resolve) => setTimeout(resolve, 800));
  const motion = await page.evaluate(() => {
    const offenders = [];
    for (const element of document.querySelectorAll('*')) {
      const style = window.getComputedStyle(element);
      const longest = (value) =>
        Math.max(
          0,
          ...value.split(',').map((part) => {
            const seconds = parseFloat(part);
            return Number.isFinite(seconds) ? seconds * (part.includes('ms') ? 0.001 : 1) : 0;
          }),
        );
      const worst = Math.max(longest(style.transitionDuration), longest(style.animationDuration));
      if (worst > 0.05) offenders.push(`${element.tagName.toLowerCase()} ${worst.toFixed(2)}s`);
    }
    return offenders.slice(0, 5);
  });
  if (motion.length > 0) {
    problems.push(`prefers-reduced-motion ignored by: ${motion.join(', ')}`);
  }
  console.log(`reduced motion ${motion.length === 0 ? 'honoured' : 'IGNORED'}`);
} finally {
  await browser.close();
}

if (problems.length > 0) {
  console.error('\nFAILED');
  for (const problem of problems) console.error(`  ${problem}`);
  exit(1);
}
console.log('\nOK');
