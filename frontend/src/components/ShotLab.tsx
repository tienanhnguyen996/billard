import React, { useState, useEffect, useRef } from 'react';
import { DrillDef } from './DrillsView';

interface LabBall {
  id: number;
  label: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  isCue: boolean;
  potted: boolean;
  color: string;
}

export const ShotLab: React.FC<{ activeDrill?: DrillDef | null }> = ({ activeDrill }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [balls, setBalls] = useState<LabBall[]>([]);
  const [power, setPower] = useState(45);
  const [aimDeg, setAimDeg] = useState(0);
  const [spinX, setSpinX] = useState(0);
  const [spinY, setSpinY] = useState(0);
  const [spinName, setSpinName] = useState('Center Ball');
  const [isSimulating, setIsSimulating] = useState(false);
  const [statusMsg, setStatusMsg] = useState('● Ready for Strike &mdash; Adjust aim &amp; spin or drag cue stick');

  const BALL_PALETTE: Record<string, string> = {
    '1': '#f59e0b', '2': '#2563eb', '3': '#dc2626', '4': '#9333ea',
    '5': '#ea580c', '6': '#16a34a', '7': '#78350f', '8': '#09090b', '9': '#eab308'
  };

  // Reset or load rack
  const loadPreset = (type: string) => {
    setIsSimulating(false);
    const newBalls: LabBall[] = [
      { id: 0, label: 'CB', x: 250, y: 250, vx: 0, vy: 0, r: 12, isCue: true, potted: false, color: '#ffffff' }
    ];

    if (type === '9ball') {
      const diamond = [
        { l: '1', dx: 0, dy: 0 },
        { l: '2', dx: 22, dy: -13 }, { l: '3', dx: 22, dy: 13 },
        { l: '4', dx: 44, dy: -26 }, { l: '9', dx: 44, dy: 0 }, { l: '5', dx: 44, dy: 26 },
        { l: '6', dx: 66, dy: -13 }, { l: '7', dx: 66, dy: 13 },
        { l: '8', dx: 88, dy: 0 }
      ];
      diamond.forEach((b, i) => {
        newBalls.push({
          id: i + 1, label: b.l, x: 680 + b.dx, y: 250 + b.dy,
          vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE[b.l] || '#ffb43c'
        });
      });
    } else if (type === '3ball') {
      newBalls.push(
        { id: 1, label: '1', x: 550, y: 180, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE['1'] },
        { id: 2, label: '2', x: 750, y: 350, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE['2'] },
        { id: 3, label: '3', x: 820, y: 150, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE['3'] }
      );
    } else if (type === 'thincut') {
      newBalls.push(
        { id: 1, label: '1', x: 780, y: 120, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE['1'] }
      );
    }

    setBalls(newBalls);
    setStatusMsg(`● Loaded ${type.toUpperCase()} Rack &mdash; Ready for Strike`);
  };

  useEffect(() => {
    if (activeDrill) {
      const newBalls: LabBall[] = [
        { id: 0, label: 'CB', x: activeDrill.cb.rx, y: activeDrill.cb.ry, vx: 0, vy: 0, r: 12, isCue: true, potted: false, color: '#ffffff' },
        { id: 1, label: '1', x: activeDrill.ob.rx, y: activeDrill.ob.ry, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE['1'] }
      ];
      (activeDrill.extraBalls || []).forEach((eb, idx) => {
        const num = String(idx + 2);
        newBalls.push({
          id: idx + 2, label: num, x: eb.rx, y: eb.ry, vx: 0, vy: 0, r: 12, isCue: false, potted: false, color: BALL_PALETTE[num] || '#ffb43c'
        });
      });
      setBalls(newBalls);
      const angle = Math.atan2(activeDrill.ob.ry - activeDrill.cb.ry, activeDrill.ob.rx - activeDrill.cb.rx);
      setAimDeg(Math.round((angle * 180 / Math.PI + 360) % 360));
      setStatusMsg(`● Loaded Drill: ${activeDrill.title}`);
    } else {
      loadPreset('9ball');
    }
  }, [activeDrill]);

  // Strike simulate
  const strikeCueBall = () => {
    if (isSimulating) return;
    const cb = balls.find(b => b.isCue && !b.potted);
    if (!cb) return;

    setIsSimulating(true);
    setStatusMsg('⚡ Simulation running &mdash; 60 FPS multi-ball physics...');

    const spd = (power / 100.0) * 22.0;
    const rad = (aimDeg * Math.PI) / 180;
    cb.vx = Math.cos(rad) * spd;
    cb.vy = Math.sin(rad) * spd;
  };

  // Physics animation loop
  useEffect(() => {
    let animId: number;

    const tick = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Update positions
      if (isSimulating) {
        const friction = 0.985;
        let anyMoving = false;

        balls.forEach(b => {
          if (b.potted) return;
          b.x += b.vx;
          b.y += b.vy;
          b.vx *= friction;
          b.vy *= friction;

          if (Math.hypot(b.vx, b.vy) > 0.05) anyMoving = true;
          else { b.vx = 0; b.vy = 0; }

          // Cushion bounces (18 to 982, 18 to 482)
          if (b.x - b.r < 18) { b.x = 18 + b.r; b.vx = -b.vx * 0.85; }
          if (b.x + b.r > 982) { b.x = 982 - b.r; b.vx = -b.vx * 0.85; }
          if (b.y - b.r < 18) { b.y = 18 + b.r; b.vy = -b.vy * 0.85; }
          if (b.y + b.r > 482) { b.y = 482 - b.r; b.vy = -b.vy * 0.85; }

          // Pockets check
          const pockets = [
            { x: 18, y: 18 }, { x: 500, y: 12 }, { x: 982, y: 18 },
            { x: 18, y: 482 }, { x: 500, y: 488 }, { x: 982, y: 482 }
          ];
          pockets.forEach(p => {
            if (Math.hypot(b.x - p.x, b.y - p.y) < 24) {
              b.potted = true;
              b.vx = 0; b.vy = 0;
            }
          });
        });

        // Ball-ball elastic collisions
        for (let i = 0; i < balls.length; i++) {
          for (let j = i + 1; j < balls.length; j++) {
            const b1 = balls[i];
            const b2 = balls[j];
            if (b1.potted || b2.potted) continue;
            const dx = b2.x - b1.x;
            const dy = b2.y - b1.y;
            const dist = Math.hypot(dx, dy);
            if (dist < b1.r + b2.r && dist > 0.001) {
              const nx = dx / dist;
              const ny = dy / dist;
              const kx = b1.vx - b2.vx;
              const ky = b1.vy - b2.vy;
              const p = 2 * (nx * kx + ny * ky) / 2;
              b1.vx -= p * nx * 0.95;
              b1.vy -= p * ny * 0.95;
              b2.vx += p * nx * 0.95;
              b2.vy += p * ny * 0.95;
              // Separate
              const overlap = (b1.r + b2.r - dist) / 2;
              b1.x -= overlap * nx;
              b1.y -= overlap * ny;
              b2.x += overlap * nx;
              b2.y += overlap * ny;
            }
          }
        }

        if (!anyMoving) {
          setIsSimulating(false);
          setStatusMsg('● Simulation complete &mdash; Ready for next strike');
        }
      }

      // Render 2D Table
      ctx.clearRect(0, 0, 1000, 500);

      // Rails & cloth
      ctx.fillStyle = '#161922';
      ctx.fillRect(0, 0, 1000, 500);
      ctx.fillStyle = '#3d4652';
      ctx.fillRect(18, 18, 964, 464);
      ctx.strokeStyle = '#282f3a';
      ctx.lineWidth = 2;
      ctx.strokeRect(18, 18, 964, 464);

      // Pockets
      [[18,18],[500,12],[982,18],[18,482],[500,488],[982,482]].forEach(p => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], 24, 0, Math.PI * 2);
        ctx.fillStyle = '#020617';
        ctx.fill();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 3;
        ctx.stroke();
      });

      // Aim Line & 90° Stun Tangent Guide (when not simulating)
      const cb = balls.find(b => b.isCue && !b.potted);
      if (cb && !isSimulating) {
        const rad = (aimDeg * Math.PI) / 180;
        const lineLen = 900;
        const targetX = cb.x + Math.cos(rad) * lineLen;
        const targetY = cb.y + Math.sin(rad) * lineLen;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(cb.x, cb.y);
        ctx.lineTo(targetX, targetY);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.75)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.stroke();

        // Check nearest intersecting object ball
        let nearestDist = 9999;
        let hitBall: LabBall | null = null;
        balls.forEach(ob => {
          if (ob.isCue || ob.potted) return;
          const dx = ob.x - cb.x;
          const dy = ob.y - cb.y;
          const proj = dx * Math.cos(rad) + dy * Math.sin(rad);
          if (proj > 0) {
            const perpDist = Math.abs(-dx * Math.sin(rad) + dy * Math.cos(rad));
            if (perpDist < cb.r + ob.r && proj < nearestDist) {
              nearestDist = proj;
              hitBall = ob;
            }
          }
        });

        if (hitBall) {
          const gbX = cb.x + Math.cos(rad) * (nearestDist - (hitBall as LabBall).r);
          const gbY = cb.y + Math.sin(rad) * (nearestDist - (hitBall as LabBall).r);
          // Ghost ball
          ctx.beginPath();
          ctx.arc(gbX, gbY, 12, 0, Math.PI * 2);
          ctx.strokeStyle = '#FFFFFF';
          ctx.stroke();

          // 90° Stun tangent line
          const perpAngle = rad + Math.PI / 2;
          ctx.beginPath();
          ctx.moveTo(gbX, gbY);
          ctx.lineTo(gbX + Math.cos(perpAngle) * 90, gbY + Math.sin(perpAngle) * 90);
          ctx.strokeStyle = '#46E6FF';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
        }
        ctx.restore();
      }

      // Draw Balls
      balls.forEach(b => {
        if (b.potted) return;
        ctx.save();
        ctx.beginPath();
        ctx.arc(b.x, b.y, b.r, 0, Math.PI * 2);
        ctx.fillStyle = b.color;
        ctx.fill();
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        if (!b.isCue) {
          ctx.fillStyle = '#fff';
          ctx.font = 'bold 8px sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(b.label, b.x, b.y);
        }
        ctx.restore();
      });

      animId = requestAnimationFrame(tick);
    };

    animId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animId);
  }, [balls, aimDeg, isSimulating]);

  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)' }}>
          Interactive Shot Lab &mdash; 2D Physics Sandbox
        </h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--mut)' }}>
          Multi-Ball Rigid-Body Physics Engine &bull; Spin &amp; Tangent Geometry Visualizer
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.4fr) minmax(320px, 0.9fr)', gap: '1.25rem' }}>
        {/* Table Canvas Side */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
          <div className="rb-card" style={{ padding: '0.85rem' }}>
            <div className="rb-radar-container">
              <canvas ref={canvasRef} width={1000} height={500} />
            </div>
          </div>

          <div className="rb-card" style={{ padding: '0.75rem 1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--spring)', fontWeight: 600 }} dangerouslySetInnerHTML={{ __html: statusMsg }} />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--cyan)' }}>
              Balls on Table: {balls.filter(b => !b.potted).length}
            </span>
          </div>
        </div>

        {/* Controls Side */}
        <div className="rb-card" style={{ padding: '1.1rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <button
            className="rb-btn active"
            style={{ width: '100%', justifyContent: 'center', padding: '0.75rem', fontSize: '0.95rem', fontWeight: 800 }}
            onClick={strikeCueBall}
          >
            🎱 STRIKE / SIMULATE SHOT!
          </button>

          {/* Preset Buttons */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            <button className="rb-btn" onClick={() => loadPreset('9ball')}>💎 9-Ball Diamond</button>
            <button className="rb-btn" onClick={() => loadPreset('3ball')}>🎯 3-Ball Position</button>
            <button className="rb-btn" onClick={() => loadPreset('thincut')}>⚡ Thin Cut 72&deg;</button>
            <button className="rb-btn" onClick={() => loadPreset('3ball')}>🔄 Reset Rack</button>
          </div>

          {/* Power Slider */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', fontWeight: 600 }}>
              <span>STRIKE POWER (SPEED)</span>
              <span style={{ color: 'var(--spring)', fontFamily: 'var(--font-mono)' }}>{power}% ({((power * 0.36)).toFixed(1)} km/h)</span>
            </div>
            <input
              type="range"
              min="10"
              max="100"
              value={power}
              onChange={(e) => setPower(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--volt)' }}
            />
          </div>

          {/* Aim Angle Slider */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', fontWeight: 600 }}>
              <span>AIM ANGLE (&deg;)</span>
              <span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>{aimDeg.toFixed(1)}&deg;</span>
            </div>
            <input
              type="range"
              min="0"
              max="360"
              step="0.5"
              value={aimDeg}
              onChange={(e) => setAimDeg(Number(e.target.value))}
              style={{ width: '100%', accentColor: 'var(--cyan)' }}
            />
          </div>

          {/* Cue Tip Spin Presets */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, background: 'var(--ink-2)', padding: '0.85rem', borderRadius: 8 }}>
            <span style={{ fontSize: '0.76rem', fontWeight: 700, color: 'var(--bone)' }}>CUE-TIP SPIN CONTACT</span>
            <div style={{ fontSize: '0.74rem', color: 'var(--volt)', fontFamily: 'var(--font-mono)' }}>{spinName}</div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              <button className="rb-btn" style={{ padding: '3px 8px', fontSize: '0.7rem' }} onClick={() => { setSpinX(0); setSpinY(0); setSpinName('Center Ball'); }}>Center</button>
              <button className="rb-btn" style={{ padding: '3px 8px', fontSize: '0.7rem' }} onClick={() => { setSpinX(0); setSpinY(-0.6); setSpinName('High Follow'); }}>High Follow</button>
              <button className="rb-btn" style={{ padding: '3px 8px', fontSize: '0.7rem' }} onClick={() => { setSpinX(0); setSpinY(0.6); setSpinName('Low Draw'); }}>Low Draw</button>
              <button className="rb-btn" style={{ padding: '3px 8px', fontSize: '0.7rem' }} onClick={() => { setSpinX(-0.6); setSpinY(0); setSpinName('Left English'); }}>Left</button>
              <button className="rb-btn" style={{ padding: '3px 8px', fontSize: '0.7rem' }} onClick={() => { setSpinX(0.6); setSpinY(0); setSpinName('Right English'); }}>Right</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
