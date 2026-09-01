# Plan: a marketing site for Vantage

A design document, not an implementation. Nothing here is built.

The research note (`frontend-craft-research.md`) ends by arguing that if the goal
is a site that wins on craft, the console is the wrong artefact to enter — a UI
optimised for an eight-hour shift is in the wrong race, and its own path to
looking exceptional is restraint. This is the other half of that argument: the
place where the full toolkit genuinely belongs, because attention *is* the
product there.

---

## 1. What it is for

One page, one job: someone who has never heard of Vantage understands within
thirty seconds what it does, believes it works, and can get to the repository.

It is not documentation. The README is 2,400 lines and does that job. It is not a
demo — a demo is `vantage run --source "synthetic://?objects=3" --dashboard`,
which anyone can run in a minute and which shows real output. This page exists to
make someone *want* to run that command.

**Audience,** in the order the page should serve them:

1. A developer evaluating the project. Wants: what it does, what it costs to run,
   what it refuses to do, where the code is.
2. Someone assessing it as a product. Wants: what problem, what evidence, what
   limits.
3. A design or engineering audience judging the work itself. Wants: craft.

The three want the same things in different proportions, which is convenient. All
three are lost by vagueness and won by specificity.

---

## 2. The constraint that shapes everything

**A marketing page for a surveillance product must not lie about surveillance.**

This is not a general principle bolted on. It is the project's own rule — the
thing the last several weeks of work has been about — and it bites hardest here,
because marketing pages are where systems are described as more capable than they
are.

Concretely, the page must not:

- **Show footage of identifiable people who did not agree to appear on it.** Every
  frame on this page is either synthetic, from a clip whose licence permits it,
  or of someone who consented. Same standard as the README's own clips.
- **Imply face recognition.** Vantage identifies nobody by default, and the
  identity subsystem is opt-in, consented-enrolment-only, and documented as
  having unverified discrimination. A hero shot of faces with names floating over
  them would be a lie about the product and about the project's position.
- **Show capabilities that only exist in the mock.** If a number appears on the
  page, it came from a run. If a chart appears, it is a real chart of real
  recorded data.
- **Use the word "AI" to mean "we did not want to explain it."** Every claim on
  the page should name the technique.

There is a positive version of this, and it is the page's best angle: **the
honesty is the pitch.** Almost nothing in this category will tell you what it
cannot do. Vantage's README has a Known Limitations section with eleven entries,
including "a walk toward the camera reads as a run" and "identity discrimination
is unverified." A marketing page that leads with that is more persuasive to
audience 1 and 3 than any amount of spectacle, and it is *differentiating* in a
way a WebGL hero is not.

---

## 3. Narrative structure

Seven beats. Scroll-driven, each a full viewport, each one claim.

| # | Beat | The claim | The visual |
| :-- | :--- | :--- | :--- |
| 1 | **Hero** | A camera does not understand what it sees. | Raw footage resolving into annotated footage — the same frames, twice |
| 2 | **The seam** | Detection is not understanding. | One frame, then the same frame with a track id, then with a state, then an activity — built up in layers |
| 3 | **Time** | The thing that matters happened *over* time. | A single entity's track drawing itself across the frame, dwell accumulating |
| 4 | **Baselines** | "Unusual" needs a definition of usual. | The real analytics chart, weeks of buckets, one anomaly capped in red |
| 5 | **Restraint** | It says when it does not know. | The console's own `Unavailable` states, quoted verbatim |
| 6 | **Limits** | Here is what it gets wrong. | Known Limitations, presented as a feature not a footnote |
| 7 | **Run it** | One command. | The command, a copy button, the repo |

Beat 6 is the one that will feel wrong to write and will be the reason anyone
remembers the page.

---

## 4. Technique per beat

Drawn from the research note's catalogue, with the ones that were wrong for a
console but right here.

**Beat 1 — Hero.** WebGL plane with a shader that mixes two video textures by a
scrubbable mask. Scroll or a drag handle moves the boundary between "raw" and
"analysed" across the same footage. This is the single highest-value effect on
the page because it *is* the product proposition, not decoration.
*three.js, custom fragment shader, one uniform driven by scroll.*

**Beat 2 — The seam.** Layer reveal choreographed with ScrollTrigger: box, then
id, then state, then activity, each pinned for a beat. The Shopify Editions
pattern — every section a narrative beat with entrance, hold, exit.

**Beat 3 — Time.** SVG path draw-on for the trail, `DrawSVG`-style, synchronised
to a dwell counter ticking up. Scroll-velocity-driven, so scrubbing back rewinds
it.

**Beat 4 — Baselines.** The real `TrendChart` component, imported from the
console. Same code, real data, no reimplementation — which is both less work and
the point.

**Beat 5 — Restraint.** Type only. After four beats of motion, a section with no
animation at all is the loudest thing on the page.

**Beat 6 — Limits.** A long scroll of plain text. Deliberately unstyled relative
to everything above it.

**Beat 7 — Run it.** Copy-to-clipboard, one link.

**Throughout:** Lenis for smooth scroll (appropriate here — this page *is* a
scroll), SplitText for headline reveals with stagger tuned to element count, View
Transitions if it becomes more than one page.

---

## 5. Stack, and a licence note that changes the answer

**The console cannot use GSAP. This page can.**

GSAP is free for commercial use since Webflow's acquisition, but the licence is
not OSI-approved, ownership stays with Webflow, and it carries no explicit grant
for redistribution inside a distributed application. Vantage ships a packaged
`vantage.exe` with the dashboard bundle inside it, under MIT. That combination is
a licence question I would not want to answer in a release, which is why the
console's craft work was done with zero new dependencies.

A marketing site is served from a URL. Nothing is redistributed, and the
restriction that matters — building a competing visual animation tool — is not
remotely in play. GSAP is the right tool here and its plugins are exactly the
ones beats 2, 3 and the headlines want.

```
Astro                 static output, islands only where interactive
three.js  (MIT)       beat 1 hero, beat 3 if it earns it
GSAP + ScrollTrigger  beat sequencing
GSAP SplitText        headline reveals
Lenis     (MIT)       scroll
Tailwind              same tokens as the console, imported not copied
```

**Reuse the console's design system rather than inventing one.** The warm
case-board palette, the type scale, `IBM Plex Mono` / `Inter` / `Source Serif 4`
— all already defined in `frontend/tailwind.config.js`. A marketing page in a
different visual language than the product it sells is a page that has to
introduce the product twice.

---

## 6. Budgets

Non-negotiable, because the research is clear that these are where award juries
take points and where real visitors leave.

| | Target | Why |
| :--- | :--- | :--- |
| Largest Contentful Paint | < 1.8s on a 4G throttle | "Slow first load" is a named jury penalty |
| Frame rate through every beat | 60fps sustained | "A hero that hitches at 40fps reads as broken, not premium" |
| Draw calls in the hero | < 100 | The three.js guidance for comfortable 60 |
| JS before the hero paints | < 150 KB gzipped | three + GSAP both load *after* first paint |
| Video | AV1/WebM with H.264 fallback, poster frame, `preload="none"` below the fold | |
| Total page weight | < 3 MB including video | |

**Mobile is not a resize.** The hero's scrub gesture has to work as a touch drag,
and the beat pinning has to degrade to plain stacked sections on a short
viewport. A poor mobile experience is described as capping the score outright,
and it is also where most of the traffic will be.

**Accessibility is a gate, not a pass.** Every beat readable and navigable with
scroll animation disabled; `prefers-reduced-motion` removes motion rather than
shortening it; visible focus on every interactive element; the video carries
captions or is decorative and marked as such; the whole narrative works as linear
text for a screen reader.

---

## 7. Shape of the work

```
site/                      separate from frontend/ — different app, different deps
├── src/pages/index.astro
├── src/beats/             one component per beat
├── src/gl/                hero shader and its loader
├── src/media/             clips, captured from real runs
└── astro.config.mjs
```

Deployed as static files. No server, no analytics that phone home — a page for a
project whose central claim is restraint about surveillance should not itself be
running third-party trackers, and saying so on the page is worth more than the
data would be.

**Rough effort**, in the order I would build it, each independently shippable:

1. Skeleton, design tokens imported, beats 5–7 as plain type — *half a day*, and
   already a legitimate one-page site.
2. Beats 2–4 with ScrollTrigger and the real chart — *1–2 days*.
3. Beat 1's WebGL hero and shader — *1–2 days*, the highest-risk item.
4. Mobile, reduced motion, performance pass, captions — *1 day*, non-optional.

The order matters: at the end of step 1 there is a real site, and every step
after improves rather than unblocks it.

---

## 8. Open questions

These want answers before step 1, not during.

1. **What footage?** The synthetic source is honest and unimpressive. The
   Wikimedia clips are real, licensed and impressive, but show identifiable
   members of the public — usable under their licences, and still worth a
   deliberate decision given what the page is about. A third option is footage of
   whoever is happy to be filmed for it.
2. **Whose project is this publicly?** Two remotes, two accounts. Attribution and
   domain follow from that.
3. **Is there a name and a domain**, or does it live at a GitHub Pages URL?
4. **Does beat 6 stay?** It is the strongest idea here and the one most likely to
   be cut. Deciding now is better than deciding when it feels risky.
