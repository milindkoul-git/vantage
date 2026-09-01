/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ─── Case Board palette ───────────────────────────────────────────────
        board: {
          DEFAULT: '#14110D',   // Cork/wood warm near-black — base background
          surface: '#1C1916',   // Slightly lighter board for panels sitting on board
          light:   '#252018',   // Hover/elevated surfaces on the board
        },
        manila: {
          DEFAULT: '#E8DCC0',   // Paper/folder panel surfaces
          dark:    '#D4C9A8',   // Slightly darker manila (e.g. active/hover states)
          darker:  '#BFB090',   // Manila shadow / folder depth
          sepia:   '#C4A882',   // Sepia-tone photograph circles
        },
        'string-red': {
          DEFAULT: '#B33A2E',   // The single accent — connections, alerts, stamps
          dark:    '#8A2B21',   // Darker variant for shadows/hover
          light:   '#C94A3C',   // Lighter variant for highlights
        },
        brass: {
          DEFAULT: '#B08D57',   // Pins, tab hardware, dividers
          light:   '#C9A96E',   // Pin head highlight
          dark:    '#8A6C3E',   // Brass shadow
        },
        'warm-white': '#E8E2D4',  // Text on dark board surfaces
        ink: {
          DEFAULT: '#1A1512',   // Text on manila/paper surfaces
          light:   '#3D2E26',   // Muted ink for secondary text on paper
          faint:   '#6B5545',   // Tertiary ink — very muted
        },
        // ─── Compat aliases (point old brand tokens at Case Board values)
        // Prevents silent invisible-text failures during transition
        brand: {
          accent:    '#B33A2E',   // was cyan — now string red
          warning:   '#B08D57',   // was amber — now brass
          alert:     '#B33A2E',   // was red — now string red (same accent)
          highlight: 'rgba(179,58,46,0.10)',
        },
        // Keep some graphite shades as board-surface shades (used in older viz components)
        graphite: {
          900: '#1C1916',
          800: '#252018',
          600: '#3D3228',
          400: '#6B5545',
          200: '#C4B898',
          100: '#E8E2D4',
        },
        // Glass aliases → transparent manila (stops hard invisible-text breakages)
        glass: {
          10:     'rgba(232,220,192,0.04)',
          20:     'rgba(232,220,192,0.08)',
          border: 'rgba(176,141,87,0.20)',
          glow:   'rgba(179,58,46,0.15)',
        },
        // Obsidian aliases → board (prevents layout breakage)
        obsidian: {
          900: '#14110D',
          800: '#1C1916',
          700: '#252018',
          600: '#2E2820',
        },
      },
      fontFamily: {
        serif: ['Source Serif 4', 'Georgia', 'serif'],
        mono:  ['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
        sans:  ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      fontSize: {
        // A real scale rather than two sizes and then Tailwind's defaults.
        //
        // Six steps on a ~1.2 ratio from 10px, which is the smallest thing this
        // interface should ever ask someone to read, and every one carries its
        // own line height and tracking - a stamped 10px label needs open
        // tracking to stay legible where an 18px serif heading needs none.
        // Juries and operators want the same thing here: consistent rhythm at
        // every breakpoint, which is impossible when half the sizes are inline.
        'micro': ['0.625rem', { lineHeight: '0.875rem', letterSpacing: '0.06em' }],
        'tiny':  ['0.75rem',  { lineHeight: '1.05rem',  letterSpacing: '0.01em' }],
        'body':  ['0.8125rem',{ lineHeight: '1.25rem',  letterSpacing: '0' }],
        'lede':  ['0.9375rem',{ lineHeight: '1.4rem',   letterSpacing: '-0.005em' }],
        'title': ['1.125rem', { lineHeight: '1.5rem',   letterSpacing: '-0.01em' }],
        'display': ['1.5rem', { lineHeight: '1.85rem',  letterSpacing: '-0.02em' }],
      },
      boxShadow: {
        // Paper drop-shadows — warm, soft, no glow
        'paper':       '0 2px 8px  0 rgba(20,17,13,0.18), 0 1px 3px 0 rgba(20,17,13,0.12)',
        'paper-lg':    '0 6px 24px 0 rgba(20,17,13,0.22), 0 2px 8px 0 rgba(20,17,13,0.14)',
        'paper-lift':  '0 12px 40px 0 rgba(20,17,13,0.28), 0 4px 12px 0 rgba(20,17,13,0.18)',
        'pin':         '0 2px 6px  0 rgba(20,17,13,0.45)',
        // Compat — old glow shadows map to subtle paper variants
        'glass-panel': '0 6px 24px 0 rgba(20,17,13,0.22)',
        'cyan-glow':   '0 2px 8px 0 rgba(179,58,46,0.18)',
        'alert-glow':  '0 2px 8px 0 rgba(179,58,46,0.22)',
      },
      backgroundImage: {
        'cork': [
          'radial-gradient(ellipse at 18% 28%, rgba(176,141,87,0.04) 0%, transparent 55%)',
          'radial-gradient(ellipse at 82% 72%, rgba(160,120,60,0.04) 0%, transparent 55%)',
          'radial-gradient(ellipse at 50% 50%, rgba(200,160,90,0.025) 0%, transparent 65%)',
        ].join(', '),
        'radial-obsidian': 'radial-gradient(circle at center, #1C1916 0%, #14110D 100%)',
      },
    },
  },
  plugins: [],
}
