import React, { useEffect, useRef } from 'react';
import { Shot, Ball } from '../types';
import { getBallShortLabel, getShotTiming } from '../utils/geometry';

interface RunTrackingPipProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  shots: Shot[];
  activeShotIndex: number;
  balls: Ball[];
  currentTime: number;
  onClose: () => void;
}

export const RunTrackingPip: React.FC<RunTrackingPipProps> = ({
  videoRef,
  shots,
  activeShotIndex,
  balls,
  currentTime,
  onClose
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let animId: number;

    const render = () => {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const W = canvas.width || 1280;
      const H = canvas.height || 720;
      ctx.clearRect(0, 0, W, H);

      // Black background
      ctx.fillStyle = '#05070D';
      ctx.fillRect(0, 0, W, H);

      // Running Makes / Misses / Run up to currentTime
      let makes = 0, misses = 0, currentRun = 0;
      shots.forEach(s => {
        const { strokeT } = getShotTiming(s);
        if (strokeT <= currentTime + 0.5) {
          const outUp = (s.outcome || '').toUpperCase();
          if (outUp.includes('MAKE') || outUp.includes('WIN')) {
            makes++;
            currentRun++;
          } else if (outUp.includes('MISS') || outUp.includes('SCRATCH')) {
            misses++;
            currentRun = 0;
          }
        }
      });

      // Scoreboard Header: MAKES X  MISSES Y  RUN Z
      ctx.font = '800 26px "Clash Display", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillStyle = '#8C93A6';
      ctx.fillText('MAKES', W * 0.32, 54);
      ctx.fillText('MISSES', W * 0.50, 54);
      ctx.fillText('RUN', W * 0.68, 54);

      ctx.font = '800 42px "JetBrains Mono", monospace';
      ctx.fillStyle = '#5BFF9E';
      ctx.fillText(String(makes), W * 0.32, 102);
      ctx.fillStyle = '#F3F5EE';
      ctx.fillText(String(misses), W * 0.50, 102);
      ctx.fillStyle = '#FEF84C';
      ctx.fillText(String(currentRun), W * 0.68, 102);

      // Centered Vector Table
      const tW = Math.round(W * 0.62);
      const tH = Math.round(tW * 0.5);
      const tX = Math.round((W - tW) * 0.40);
      const tY = Math.round((H - tH) * 0.56);

      ctx.fillStyle = '#0C1020';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(tX - 16, tY - 16, tW + 32, tH + 32, 14) : ctx.rect(tX - 16, tY - 16, tW + 32, tH + 32);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.22)';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#11172B';
      ctx.fillRect(tX, tY, tW, tH);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 2;
      ctx.strokeRect(tX, tY, tW, tH);

      const mapVX = (rx: number) => tX + (Math.max(0, Math.min(1000, rx)) / 1000) * tW;
      const mapVY = (ry: number) => tY + (Math.max(0, Math.min(500, ry)) / 500) * tH;

      // Pockets
      [[0,0],[500,0],[1000,0],[0,500],[500,500],[1000,500]].forEach(p => {
        ctx.beginPath();
        ctx.arc(mapVX(p[0]), mapVY(p[1]), 14, 0, Math.PI * 2);
        ctx.fillStyle = '#05070D';
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Active Shot trajectories
      const curShot = shots[activeShotIndex] || shots[0];
      if (curShot) {
        if (curShot.cue_path && curShot.cue_path.length > 1) {
          ctx.beginPath();
          ctx.moveTo(mapVX(curShot.cue_path[0].rx), mapVY(curShot.cue_path[0].ry));
          curShot.cue_path.forEach(pt => ctx.lineTo(mapVX(pt.rx), mapVY(pt.ry)));
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 3;
          ctx.setLineDash([6, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        if (curShot.obj_path && curShot.obj_path.length > 1) {
          ctx.beginPath();
          ctx.moveTo(mapVX(curShot.obj_path[0].rx), mapVY(curShot.obj_path[0].ry));
          curShot.obj_path.forEach(pt => ctx.lineTo(mapVX(pt.rx), mapVY(pt.ry)));
          ctx.strokeStyle = '#FEF84C';
          ctx.lineWidth = 3.5;
          ctx.stroke();
        }
      }

      // Balls
      const targetTag = curShot ? getBallShortLabel(curShot.target_ball || '') : '';
      balls.forEach(b => {
        const tag = getBallShortLabel(b.label);
        const isCue = tag === 'CB';
        const isTarget = tag === targetTag;
        ctx.beginPath();
        ctx.arc(mapVX(b.rx), mapVY(b.ry), 9, 0, Math.PI * 2);
        ctx.fillStyle = isCue ? '#FFFFFF' : (isTarget ? '#FEF84C' : '#495057');
        ctx.fill();
        ctx.strokeStyle = '#05070D';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Bottom-Right Picture-in-Picture (PiP)
      if (video) {
        const pipW = Math.round(W * 0.28);
        const pipH = Math.round(H * 0.28);
        const pipX = W - pipW - 28;
        const pipY = H - pipH - 24;

        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.85)';
        ctx.shadowBlur = 18;
        ctx.fillStyle = '#000';
        ctx.fillRect(pipX - 2, pipY - 2, pipW + 4, pipH + 4);
        try {
          ctx.drawImage(video, pipX, pipY, pipW, pipH);
        } catch (_) {}
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2.5;
        ctx.strokeRect(pipX, pipY, pipW, pipH);
        ctx.restore();
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [shots, activeShotIndex, balls, currentTime, videoRef]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#000' }}>
      <canvas ref={canvasRef} width={1280} height={720} style={{ width: '100%', height: '100%', display: 'block' }} />
      <button
        className="rb-btn"
        style={{ position: 'absolute', top: 16, right: 16, zIndex: 20 }}
        onClick={onClose}
      >
        ✕ Exit PiP Mode
      </button>
    </div>
  );
};
