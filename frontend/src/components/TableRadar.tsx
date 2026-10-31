import React, { useEffect, useRef } from 'react';
import { Shot, Ball } from '../types';
import { BALL_COLORS, bgrToCss, getBallShortLabel, getShotTiming } from '../utils/geometry';

interface TableRadarProps {
  shot?: Shot;
  balls: Ball[];
  clothMode?: 'auto' | 'shark' | 'blue' | 'green';
  sessionMode?: string;
  currentTime?: number;
}

export const TableRadar: React.FC<TableRadarProps> = ({
  shot,
  balls,
  clothMode = 'auto',
  sessionMode = 'hs',
  currentTime = 0
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, 1000, 500);

    // 1. Cloth & Rails Style
    let clothBg = '#3d4652'; // Simonis 860 Shark Grey
    let railsBg = '#161922';
    let cushionBorder = '#282f3a';

    if (clothMode === 'blue' || (clothMode === 'auto' && sessionMode !== 'hs')) {
      clothBg = '#0c4a6e'; // Tournament Blue
      railsBg = '#082f49';
      cushionBorder = '#0284c7';
    } else if (clothMode === 'green') {
      clothBg = '#0a4d33';
      railsBg = '#1c2d22';
      cushionBorder = '#14532d';
    }

    // Draw rails
    ctx.fillStyle = railsBg;
    ctx.fillRect(0, 0, 1000, 500);

    // Draw cloth
    ctx.fillStyle = clothBg;
    ctx.fillRect(18, 18, 964, 464);

    // Cushion border
    ctx.strokeStyle = cushionBorder;
    ctx.lineWidth = 2;
    ctx.strokeRect(18, 18, 964, 464);

    // Diamond sights
    const drawDiamond = (x: number, y: number) => {
      ctx.save();
      ctx.fillStyle = '#e2e8f0';
      ctx.beginPath();
      ctx.moveTo(x, y - 3.5);
      ctx.lineTo(x + 3.5, y);
      ctx.lineTo(x, y + 3.5);
      ctx.lineTo(x - 3.5, y);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    };

    [140, 260, 380, 620, 740, 860].forEach(x => {
      drawDiamond(x, 9);
      drawDiamond(x, 491);
    });
    [90, 170, 250, 330, 410].forEach(y => {
      drawDiamond(9, y);
      drawDiamond(991, y);
    });

    // Markings
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.16)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(250, 18); ctx.lineTo(250, 482);
    ctx.moveTo(750, 18); ctx.lineTo(750, 482);
    ctx.stroke();
    ctx.setLineDash([]);
    [[250, 250], [500, 250], [750, 250]].forEach(pt => {
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], 3.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.fill();
    });
    ctx.restore();

    // 6 Pockets
    const pockets = [
      { rx: 18, ry: 18 }, { rx: 500, ry: 12 }, { rx: 982, ry: 18 },
      { rx: 18, ry: 482 }, { rx: 500, ry: 488 }, { rx: 982, ry: 482 }
    ];
    pockets.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.rx, p.ry, 24, 0, Math.PI * 2);
      ctx.fillStyle = '#020617';
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#334155';
      ctx.stroke();
    });

    const mapRx = (rx: number) => 18 + (Math.max(0, Math.min(1000, rx)) / 1000.0) * 964.0;
    const mapRy = (ry: number) => 18 + (Math.max(0, Math.min(500, ry)) / 500.0) * 464.0;

    // Shot Trajectories & Aim Line on 2D Table
    if (shot) {
      const isBreakShot = shot.shot_number === 1 || (shot.shot_type && shot.shot_type.toLowerCase().includes('break'));

      // Cue Ball Path / Aim Line
      if (shot.cue_path && shot.cue_path.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(mapRx(shot.cue_path[0].rx), mapRy(shot.cue_path[0].ry));
        for (let i = 1; i < shot.cue_path.length; i++) {
          ctx.lineTo(mapRx(shot.cue_path[i].rx), mapRy(shot.cue_path[i].ry));
        }
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#ffffff';
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.restore();
      } else if (shot.aim_line) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(mapRx(shot.aim_line.cb_rx), mapRy(shot.aim_line.cb_ry));
        ctx.lineTo(mapRx(shot.aim_line.gb_rx), mapRy(shot.aim_line.gb_ry));
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
        ctx.lineWidth = 2.5;
        ctx.setLineDash([6, 4]);
        ctx.stroke();
        ctx.restore();
      }

      // Object Ball Path
      const outStr = (shot.outcome || '').toUpperCase();
      const trajColor = outStr.includes('MAKE') ? '#10b981' : outStr.includes('SAFETY') ? '#f59e0b' : '#ef4444';

      if (shot.obj_path && shot.obj_path.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(mapRx(shot.obj_path[0].rx), mapRy(shot.obj_path[0].ry));
        for (let i = 1; i < shot.obj_path.length; i++) {
          ctx.lineTo(mapRx(shot.obj_path[i].rx), mapRy(shot.obj_path[i].ry));
        }
        ctx.strokeStyle = trajColor;
        ctx.lineWidth = 3;
        ctx.shadowColor = trajColor;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.restore();
      } else if (!isBreakShot && shot.aim_line) {
        const obX = shot.aim_line.ob_rx !== undefined ? shot.aim_line.ob_rx : shot.aim_line.gb_rx;
        const obY = shot.aim_line.ob_ry !== undefined ? shot.aim_line.ob_ry : shot.aim_line.gb_ry;
        const potDist = Math.hypot(shot.aim_line.pock_rx - obX, shot.aim_line.pock_ry - obY);
        if (potDist > 15) {
          ctx.save();
          ctx.beginPath();
          ctx.moveTo(mapRx(obX), mapRy(obY));
          ctx.lineTo(mapRx(shot.aim_line.pock_rx), mapRy(shot.aim_line.pock_ry));
          ctx.strokeStyle = trajColor;
          ctx.lineWidth = 3;
          ctx.stroke();
          ctx.restore();
        }
      }

      // Ghost Ball & Cut Angle Badge (Only for non-break shots with cut angle)
      if (shot.aim_line && !isBreakShot) {
        const al = shot.aim_line;
        const gbX = mapRx(al.gb_rx), gbY = mapRy(al.gb_ry);
        ctx.save();
        ctx.beginPath();
        ctx.arc(gbX, gbY, 11, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.8;
        ctx.setLineDash([4, 3]);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.fill();
        ctx.stroke();

        if (shot.cut_angle_deg !== undefined && shot.cut_angle_deg > 0) {
          const badgeX = Math.min(930, Math.max(24, gbX + 14));
          const badgeY = Math.min(470, Math.max(28, gbY - 14));
          ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
          ctx.beginPath();
          ctx.roundRect ? ctx.roundRect(badgeX - 4, badgeY - 10, 52, 16, 4) : ctx.rect(badgeX - 4, badgeY - 10, 52, 16);
          ctx.fill();
          ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.fillStyle = '#fbbf24';
          ctx.font = 'bold 9px sans-serif';
          ctx.fillText(`${shot.cut_angle_deg}°`, badgeX + 4, badgeY + 2);
        }
        ctx.restore();
      }
    }

    // Realistic Billiard Balls
    balls.forEach(b => {
      const isCue = b.label.toLowerCase().includes('cue');
      const tag = getBallShortLabel(b.label);
      const baseColor = isCue ? '#ffffff' : (BALL_COLORS[tag] || bgrToCss(b.color) || '#ffb43c');
      const bx = mapRx(b.rx);
      const by = mapRy(b.ry);

      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.6)';
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 2;
      ctx.shadowOffsetY = 3;

      const grad = ctx.createRadialGradient(bx - 3, by - 3, 1, bx, by, 12);
      if (isCue) {
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.5, '#f8fafc');
        grad.addColorStop(1, '#cbd5e1');
      } else {
        grad.addColorStop(0, '#ffffff');
        grad.addColorStop(0.3, baseColor);
        grad.addColorStop(1, '#0f172a');
      }

      ctx.beginPath();
      ctx.arc(bx, by, 12, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.stroke();

      if (!isCue) {
        ctx.beginPath();
        ctx.arc(bx, by, 5.2, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 8px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(tag, bx, by);
      }
      ctx.restore();
    });
  }, [shot, balls, clothMode, sessionMode, currentTime]);

  return (
    <div className="rb-radar-container">
      <canvas ref={canvasRef} width={1000} height={500} />
    </div>
  );
};
