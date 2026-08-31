/**
 * Top-down plot of where entities are on the facility floor.
 *
 * It plots `x` and `y` in metres, fitted to the camera footprints the radar
 * reports, plus each entity's recorded trail.
 *
 * The version this replaces read the entity list for its length and then placed
 * every dot at `sin(index * 1.5 + Date.now() * 0.001)` -- dots orbiting the
 * centre on a timer, ignoring the coordinates entirely. It looked like a working
 * radar and tracked nothing.
 */

import React, { useEffect, useRef } from 'react';
import type { RadarResponse } from '../../contracts/types';

const shortId = (id: string) => id.replace(/^global_person_/i, 'P-').replace(/^person_/i, 'P-');

export const FloorplanRadar2D: React.FC<{ radar: RadarResponse }> = ({ radar }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const zones = radar.zones ?? [];
    const entities = radar.entities ?? [];

    const draw = () => {
      const parent = canvas.parentElement;
      const cssWidth = parent?.clientWidth || 320;
      const cssHeight = parent?.clientHeight || 240;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = cssWidth * ratio;
      canvas.height = cssHeight * ratio;
      canvas.style.width = `${cssWidth}px`;
      canvas.style.height = `${cssHeight}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

      ctx.fillStyle = '#14110D';
      ctx.fillRect(0, 0, cssWidth, cssHeight);

      // Fit to whatever the facility actually spans, so the plot is to scale
      // rather than to a guessed extent.
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;
      const consider = (x: number, y: number) => {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      };
      for (const zone of zones) {
        consider(zone.rect[0], zone.rect[1]);
        consider(zone.rect[2], zone.rect[3]);
      }
      for (const entity of entities) consider(entity.x, entity.y);
      if (!Number.isFinite(minX)) {
        ctx.fillStyle = '#6B5545';
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillText('No footprint or position reported', 12, 20);
        return;
      }

      const pad = 18;
      const spanX = Math.max(1, maxX - minX);
      const spanY = Math.max(1, maxY - minY);
      const scale = Math.min((cssWidth - pad * 2) / spanX, (cssHeight - pad * 2) / spanY);
      const offsetX = pad + (cssWidth - pad * 2 - spanX * scale) / 2;
      const offsetY = pad + (cssHeight - pad * 2 - spanY * scale) / 2;
      const px = (x: number) => offsetX + (x - minX) * scale;
      const py = (y: number) => offsetY + (y - minY) * scale;

      for (const zone of zones) {
        const [x0, y0, x1, y1] = zone.rect;
        ctx.strokeStyle = 'rgba(176,141,87,0.28)';
        ctx.lineWidth = 1;
        ctx.strokeRect(px(x0), py(y0), (x1 - x0) * scale, (y1 - y0) * scale);
        ctx.fillStyle = 'rgba(176,141,87,0.55)';
        ctx.font = '9px ui-monospace, monospace';
        ctx.fillText(zone.name, px(x0) + 4, py(y0) + 11);

        ctx.beginPath();
        ctx.arc(px(zone.origin[0]), py(zone.origin[1]), 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#B08D57';
        ctx.fill();
      }

      for (const entity of entities) {
        if (entity.trail.length > 1) {
          ctx.beginPath();
          ctx.moveTo(px(entity.trail[0][0]), py(entity.trail[0][1]));
          for (const [tx, ty] of entity.trail.slice(1)) ctx.lineTo(px(tx), py(ty));
          ctx.strokeStyle = 'rgba(176,141,87,0.45)';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }

        const x = px(entity.x);
        const y = py(entity.y);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = entity.motion === 'stationary' ? '#B08D57' : '#B33A2E';
        ctx.fill();
        ctx.strokeStyle = '#E8DCC0';
        ctx.lineWidth = 1.2;
        ctx.stroke();

        ctx.fillStyle = '#E8DCC0';
        ctx.font = '700 9px ui-monospace, monospace';
        ctx.fillText(shortId(entity.id), x + 7, y + 3);
      }

      ctx.fillStyle = 'rgba(107,85,69,0.9)';
      ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(`${spanX.toFixed(0)}m × ${spanY.toFixed(0)}m`, 6, cssHeight - 6);
    };

    draw();
    // Redraw on resize only. The data itself arrives on the polling interval,
    // and re-rendering a static plot at 60fps would burn a core for nothing.
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [radar]);

  return <canvas ref={canvasRef} className="block h-full w-full" />;
};
