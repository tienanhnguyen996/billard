import { Shot, Ball } from '../types';

export const VERDICT_GLYPHS = {
  Brilliant: { glyph: '!!', background: '#1BAAA0', label: 'Ginned', description: 'Exceptional make' },
  Great:     { glyph: '!',  background: '#2F8EF5', label: 'Great', description: 'Difficult make' },
  Mistake:   { glyph: '?!', background: '#E6A23C', label: 'Inaccurate', description: 'Makeable miss' },
  Blunder:   { glyph: '??', background: '#D64545', label: 'Dogged', description: 'Easy miss' },
  Standard:  { glyph: '✓',  background: '#10b981', label: 'Solid Make', description: 'Controlled pot' },
  Safety:    { glyph: '🛡️', background: '#FFA726', label: 'Safety', description: 'Tactical leave' }
};

export function getShotVerdict(shot?: Shot) {
  if (!shot) return VERDICT_GLYPHS.Standard;
  const outUp = (shot.outcome || '').toUpperCase();
  const diff = Number(shot.difficulty_score || 40);
  const grade = (shot.grade || 'B').toUpperCase();
  if (outUp.includes('SAFETY')) return VERDICT_GLYPHS.Safety;
  if (outUp.includes('MAKE') || outUp.includes('WIN')) {
    if (shot.is_highlight || grade === 'S' || diff >= 58) return VERDICT_GLYPHS.Brilliant;
    if (grade === 'A' || diff >= 44) return VERDICT_GLYPHS.Great;
    return VERDICT_GLYPHS.Standard;
  }
  if (diff <= 38) return VERDICT_GLYPHS.Blunder;
  return VERDICT_GLYPHS.Mistake;
}

export function getShotTiming(shot?: Shot) {
  if (!shot) return { strokeT: 0, impactT: 0, stopT: 0 };
  const strokeT = shot.stroke_time !== undefined && shot.stroke_time !== null
    ? shot.stroke_time
    : (shot.cue_path && shot.cue_path.length > 0 ? shot.cue_path[0].t : shot.start_time);
  const impactT = shot.impact_time !== undefined && shot.impact_time !== null
    ? shot.impact_time
    : (shot.obj_path && shot.obj_path.length > 0 ? shot.obj_path[0].t : (strokeT + 0.35));
  const stopT = shot.stop_time !== undefined && shot.stop_time !== null
    ? shot.stop_time
    : (shot.cue_path && shot.cue_path.length > 0 ? shot.cue_path[shot.cue_path.length - 1].t : shot.end_time);
  return { strokeT, impactT, stopT };
}

export function getBallShortLabel(label?: string): string {
  if (!label) return '';
  const lower = label.toLowerCase();
  if (lower.includes('cue')) return 'CB';
  const m = label.match(/(\d+)/);
  if (m) return m[1];
  return label.slice(0, 2);
}

export const BALL_COLORS: Record<string, string> = {
  '1': '#f59e0b',
  '2': '#2563eb',
  '3': '#dc2626',
  '4': '#9333ea',
  '5': '#ea580c',
  '6': '#16a34a',
  '7': '#78350f',
  '8': '#09090b',
  '9': '#eab308',
  'CB': '#ffffff'
};

export function bgrToCss(bgr?: any, alpha = 1): string {
  if (Array.isArray(bgr) && bgr.length >= 3) {
    return `rgba(${bgr[2]}, ${bgr[1]}, ${bgr[0]}, ${alpha})`;
  }
  if (typeof bgr === 'string') return bgr;
  return `rgba(56, 189, 248, ${alpha})`;
}

/**
 * Mathematically exact 2D shotline renderer for the Video Overlay
 */
export function drawVideoOverlayShotline(
  ctx: CanvasRenderingContext2D,
  shot: Shot,
  balls: Ball[],
  curTime: number,
  isPaused: boolean
) {
  const { strokeT, impactT, stopT } = getShotTiming(shot);
  const isBreakShot = shot.shot_number === 1 || (shot.shot_type && shot.shot_type.toLowerCase().includes('break'));

  // Clean timing phase resolution:
  // Aiming phase: curTime < strokeT (during pre-shot aim or when paused before stroke)
  // Stroke in flight: curTime >= strokeT
  // Impact and roll: curTime >= impactT
  const isAimingPhase = curTime < strokeT;
  const isStrokePhase = curTime >= strokeT;
  const isImpactPhase = curTime >= impactT;

  const outStr = (shot.outcome || '').toUpperCase();
  const shotColor = outStr.includes('MAKE') ? '#10b981' : outStr.includes('SAFETY') ? '#f59e0b' : '#ef4444';

  const al = shot.aim_line;
  if (!al) return;

  // 1. AIMING PHASE: Crisp dashed Ghost Ball aiming line
  if (isAimingPhase) {
    ctx.save();

    // A. Cue Ball approach line to Ghost Ball (straight dashed guide)
    ctx.beginPath();
    ctx.moveTo(al.cb_x, al.cb_y);
    ctx.lineTo(al.gb_x, al.gb_y);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
    ctx.lineWidth = 2.0;
    ctx.setLineDash([6, 4]);
    ctx.stroke();

    // B. Hollow dashed Ghost Ball circle at impact position
    ctx.beginPath();
    ctx.arc(al.gb_x, al.gb_y, 9.5, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.lineWidth = 1.6;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.fill();
    ctx.stroke();

    // C & D. Object Ball to Pocket target line + Stun Tangent line
    // For break shots, the cue ball aims straight into the head ball of the rack (no pocket line or stun tangent).
    if (!isBreakShot) {
      const startX = al.ob_x !== undefined ? al.ob_x : al.gb_x;
      const startY = al.ob_y !== undefined ? al.ob_y : al.gb_y;
      const potDist = Math.hypot(al.pock_x - startX, al.pock_y - startY);

      if (potDist > 15) {
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(al.pock_x, al.pock_y);
        ctx.strokeStyle = shotColor;
        ctx.lineWidth = 2.0;
        ctx.setLineDash([]);
        ctx.stroke();

        // 90° Stun Tangent Deflection Line from Ghost Ball
        const perpX = -(al.pock_y - startY) / potDist;
        const perpY = (al.pock_x - startX) / potDist;
        const appDx = al.gb_x - al.cb_x;
        const appDy = al.gb_y - al.cb_y;
        const sign = (perpX * appDx + perpY * appDy) < 0 ? -1 : 1;
        const tangentLen = 35.0;

        ctx.beginPath();
        ctx.moveTo(al.gb_x, al.gb_y);
        ctx.lineTo(al.gb_x + perpX * sign * tangentLen, al.gb_y + perpY * sign * tangentLen);
        ctx.strokeStyle = 'rgba(70, 230, 255, 0.75)';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  // 2. LIVE STROKE / ROLLING PHASE: Smooth continuous trails (no polygon leaps)
  if (isStrokePhase && shot.cue_path && shot.cue_path.length > 0) {
    ctx.save();
    const pts = (curTime >= stopT)
      ? shot.cue_path
      : shot.cue_path.filter(pt => pt.t <= curTime + 0.08);

    if (pts.length > 1) {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = '#ffffff';
      ctx.shadowBlur = 6;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Object ball rolling trail
  if (isImpactPhase && shot.obj_path && shot.obj_path.length > 0) {
    ctx.save();
    const pts = (curTime >= stopT)
      ? shot.obj_path
      : shot.obj_path.filter(pt => pt.t <= curTime + 0.08);

    if (pts.length > 1) {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(pts[i].x, pts[i].y);
      }
      ctx.strokeStyle = shotColor;
      ctx.lineWidth = 2.5;
      ctx.shadowColor = shotColor;
      ctx.shadowBlur = 6;
      ctx.stroke();
    }
    ctx.restore();
  }
}

/**
 * 2D Vector Shot Diagram for Cards & Preview
 */
export function drawMiniShotDiagram(
  canvas: HTMLCanvasElement,
  shot: Shot,
  sampleBalls: Ball[] = []
) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const W = canvas.width || 280;
  const H = canvas.height || 128;
  ctx.clearRect(0, 0, W, H);

  const padX = 12, padY = 10;
  const tW = W - padX * 2, tH = H - padY * 2;

  // Dark vector table
  ctx.fillStyle = '#05070D';
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(padX, padY, tW, tH, 6) : ctx.rect(padX, padY, tW, tH);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
  ctx.lineWidth = 1.2;
  ctx.stroke();

  const mapX = (rx: number) => padX + (Math.max(0, Math.min(1000, rx)) / 1000) * tW;
  const mapY = (ry: number) => padY + (Math.max(0, Math.min(500, ry)) / 500) * tH;

  // Pockets
  const pockets = [
    { rx: 18, ry: 18 }, { rx: 500, ry: 12 }, { rx: 982, ry: 18 },
    { rx: 18, ry: 482 }, { rx: 500, ry: 488 }, { rx: 982, ry: 482 }
  ];
  const targetPockX = shot.aim_line ? shot.aim_line.pock_rx : -999;
  const targetPockY = shot.aim_line ? shot.aim_line.pock_ry : -999;
  const isMake = (shot.outcome || '').toUpperCase().includes('MAKE');

  pockets.forEach(p => {
    const isTarget = Math.hypot(p.rx - targetPockX, p.ry - targetPockY) < 60;
    ctx.beginPath();
    ctx.arc(mapX(p.rx), mapY(p.ry), isTarget ? 6.5 : 4.0, 0, Math.PI * 2);
    ctx.fillStyle = '#05070D';
    ctx.fill();
    ctx.lineWidth = isTarget ? 2.0 : 1.0;
    ctx.strokeStyle = isTarget ? (isMake ? '#5BFF9E' : '#FFA726') : 'rgba(255,255,255,0.2)';
    ctx.stroke();
  });

  // Background balls
  const targetTag = getBallShortLabel(shot.target_ball || '');
  sampleBalls.forEach(b => {
    const bTag = getBallShortLabel(b.label);
    if (bTag === 'CB' || bTag === targetTag) return;
    ctx.beginPath();
    ctx.arc(mapX(b.rx), mapY(b.ry), 3.4, 0, Math.PI * 2);
    ctx.fillStyle = '#495057';
    ctx.fill();
  });

  // Cue ball path
  if (shot.cue_path && shot.cue_path.length > 1) {
    const pts = shot.cue_path;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(mapX(pts[0].rx), mapY(pts[0].ry));
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(mapX(pts[i].rx), mapY(pts[i].ry));
    }
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    // Solid start CB
    ctx.beginPath();
    ctx.arc(mapX(pts[0].rx), mapY(pts[0].ry), 4.0, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.restore();
  } else if (shot.aim_line) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(mapX(shot.aim_line.cb_rx), mapY(shot.aim_line.cb_ry));
    ctx.lineTo(mapX(shot.aim_line.gb_rx), mapY(shot.aim_line.gb_ry));
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.6;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(mapX(shot.aim_line.cb_rx), mapY(shot.aim_line.cb_ry), 4.0, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();
    ctx.restore();
  }

  // Object ball path
  const isBreakShot = shot.shot_number === 1 || (shot.shot_type && shot.shot_type.toLowerCase().includes('break'));
  if (shot.obj_path && shot.obj_path.length > 1) {
    const pts = shot.obj_path;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(mapX(pts[0].rx), mapY(pts[0].ry));
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(mapX(pts[i].rx), mapY(pts[i].ry));
    }
    ctx.strokeStyle = '#FEF84C';
    ctx.lineWidth = 2.0;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(mapX(pts[0].rx), mapY(pts[0].ry), 4.2, 0, Math.PI * 2);
    ctx.fillStyle = '#FEF84C';
    ctx.fill();
    ctx.restore();
  } else if (shot.aim_line) {
    ctx.save();
    const obX = shot.aim_line.ob_rx !== undefined ? shot.aim_line.ob_rx : shot.aim_line.gb_rx;
    const obY = shot.aim_line.ob_ry !== undefined ? shot.aim_line.ob_ry : shot.aim_line.gb_ry;
    const potDist = Math.hypot(shot.aim_line.pock_rx - obX, shot.aim_line.pock_ry - obY);
    if (!isBreakShot && potDist > 15) {
      ctx.beginPath();
      ctx.moveTo(mapX(obX), mapY(obY));
      ctx.lineTo(mapX(shot.aim_line.pock_rx), mapY(shot.aim_line.pock_ry));
      ctx.strokeStyle = '#FEF84C';
      ctx.lineWidth = 2.0;
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(mapX(obX), mapY(obY), 4.2, 0, Math.PI * 2);
    ctx.fillStyle = '#FEF84C';
    ctx.fill();
    ctx.restore();
  }
}
