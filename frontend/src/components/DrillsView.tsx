import React, { useEffect, useRef } from 'react';

export interface DrillDef {
  id: string;
  title: string;
  category: string;
  shots: number;
  makePct: string;
  bestRun: number;
  desc: string;
  cb: { rx: number; ry: number };
  ob: { rx: number; ry: number };
  pock: { rx: number; ry: number };
  leave: { rx: number; ry: number };
  extraBalls?: Array<{ rx: number; ry: number }>;
}

export const DRILLS_LIST: DrillDef[] = [
  {
    id: 'x_drill',
    title: 'X Drill',
    category: 'Position & Tangent Control',
    shots: 15,
    makePct: '87%',
    bestRun: 13,
    desc: 'Diagonal criss-cross object balls forming an X across the center spot. Demands pure stun and controlled draw to hold center-table position.',
    cb: { rx: 260, ry: 360 },
    ob: { rx: 500, ry: 250 },
    pock: { rx: 982, ry: 18 },
    leave: { rx: 460, ry: 360 },
    extraBalls: [{ rx: 350, ry: 150 }, { rx: 650, ry: 150 }, { rx: 350, ry: 350 }, { rx: 650, ry: 350 }]
  },
  {
    id: 'spot_shot',
    title: 'Spot Shot Drill',
    category: 'Shot-Making Fundamentals',
    shots: 10,
    makePct: '80%',
    bestRun: 8,
    desc: 'Object ball frozen on the foot spot with cue ball in the kitchen behind the head string. Tests dead-straight delivery and ghost-ball visualization.',
    cb: { rx: 240, ry: 340 },
    ob: { rx: 750, ry: 250 },
    pock: { rx: 982, ry: 18 },
    leave: { rx: 820, ry: 320 }
  },
  {
    id: 'long_rail_cut',
    title: 'Long Rail Cut Drill',
    category: 'Thin Cuts & Cushion Sighting',
    shots: 12,
    makePct: '75%',
    bestRun: 9,
    desc: 'Object ball 1 diamond off the long rail, cue ball near center table. Develops precision aiming on 35°–55° cuts down the rail.',
    cb: { rx: 320, ry: 260 },
    ob: { rx: 640, ry: 75 },
    pock: { rx: 982, ry: 18 },
    leave: { rx: 720, ry: 240 },
    extraBalls: [{ rx: 480, ry: 75 }, { rx: 800, ry: 75 }]
  },
  {
    id: 'long_rail_back_cut',
    title: 'Long Rail Back Cut Drill',
    category: 'Advanced Inside/Outside Spin',
    shots: 10,
    makePct: '70%',
    bestRun: 7,
    desc: 'Steep >52° back-cut into the corner pocket with natural 2-rail cue ball breakout back to center table.',
    cb: { rx: 620, ry: 280 },
    ob: { rx: 740, ry: 110 },
    pock: { rx: 982, ry: 18 },
    leave: { rx: 500, ry: 410 }
  },
  {
    id: 'short_rail_cut',
    title: 'Short Rail Cut Drill',
    category: 'End-Rail Speed Control',
    shots: 10,
    makePct: '90%',
    bestRun: 10,
    desc: 'Object ball along the foot rail with delicate Gear 2 speed to hold the cue ball for the next corner pot.',
    cb: { rx: 560, ry: 310 },
    ob: { rx: 860, ry: 250 },
    pock: { rx: 982, ry: 18 },
    leave: { rx: 740, ry: 180 },
    extraBalls: [{ rx: 860, ry: 360 }]
  }
];

const DrillCanvas: React.FC<{ drill: DrillDef }> = ({ drill }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Brown wood rails
    ctx.fillStyle = '#4A2C18';
    ctx.beginPath();
    ctx.rect(0, 0, W, H);
    ctx.fill();

    // Tournament blue cloth
    const pad = 12;
    const tW = W - pad * 2;
    const tH = H - pad * 2;
    ctx.fillStyle = '#1565C0';
    ctx.fillRect(pad, pad, tW, tH);
    ctx.strokeStyle = '#0D47A1';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(pad, pad, tW, tH);

    const mapX = (rx: number) => pad + (rx / 1000) * tW;
    const mapY = (ry: number) => pad + (ry / 500) * tH;

    // 6 Pockets
    [[0,0],[500,0],[1000,0],[0,500],[500,500],[1000,500]].forEach(p => {
      const isTarget = Math.hypot(p[0] - drill.pock.rx, p[1] - drill.pock.ry) < 90;
      ctx.beginPath();
      ctx.arc(mapX(p[0]), mapY(p[1]), isTarget ? 7 : 4.5, 0, Math.PI * 2);
      ctx.fillStyle = '#05070D';
      ctx.fill();
      ctx.lineWidth = isTarget ? 2 : 1;
      ctx.strokeStyle = isTarget ? '#5BFF9E' : '#263238';
      ctx.stroke();
    });

    // Extra balls
    (drill.extraBalls || []).forEach(b => {
      ctx.beginPath();
      ctx.arc(mapX(b.rx), mapY(b.ry), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(254, 248, 76, 0.45)';
      ctx.fill();
      ctx.strokeStyle = '#000';
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // CB approach line to OB
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(mapX(drill.cb.rx), mapY(drill.cb.ry));
    ctx.lineTo(mapX(drill.ob.rx), mapY(drill.ob.ry));
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 1.8;
    ctx.stroke();

    // OB to pocket line
    ctx.beginPath();
    ctx.moveTo(mapX(drill.ob.rx), mapY(drill.ob.ry));
    ctx.lineTo(mapX(drill.pock.rx), mapY(drill.pock.ry));
    ctx.strokeStyle = '#FEF84C';
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Cue ball
    ctx.beginPath();
    ctx.arc(mapX(drill.cb.rx), mapY(drill.cb.ry), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#FFFFFF';
    ctx.fill();

    // Object ball
    ctx.beginPath();
    ctx.arc(mapX(drill.ob.rx), mapY(drill.ob.ry), 5, 0, Math.PI * 2);
    ctx.fillStyle = '#FEF84C';
    ctx.fill();
    ctx.restore();
  }, [drill]);

  return <canvas ref={canvasRef} width={320} height={150} style={{ width: '100%', height: 140, borderRadius: 8, display: 'block' }} />;
};

export const DrillsView: React.FC<{ onLaunchDrill: (drill: DrillDef) => void }> = ({ onLaunchDrill }) => {
  return (
    <div style={{ padding: '1.4rem', maxWidth: 1080, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)' }}>
          Shot-Making &amp; Position Drills
        </h2>
        <p style={{ fontSize: '0.82rem', color: 'var(--mut)' }}>
          Official Railbird Drills &bull; Click &ldquo;Start Drill&rdquo; to load directly into the 2D Physics Shot Lab
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}>
        {DRILLS_LIST.map((drill) => (
          <div key={drill.id} className="rb-card" style={{ padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--bone)' }}>{drill.title}</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--volt)', fontWeight: 600 }}>{drill.category}</div>
              </div>
              <span className="rb-glyph" style={{ background: '#1BAAA0' }}>!!</span>
            </div>

            <DrillCanvas drill={drill} />

            <p style={{ fontSize: '0.78rem', color: 'var(--mut)', lineHeight: 1.45 }}>{drill.desc}</p>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--panel-border)', paddingTop: '0.6rem', marginTop: 'auto' }}>
              <div style={{ display: 'flex', gap: 12, fontFamily: 'var(--font-mono)', fontSize: '0.76rem' }}>
                <span>Shots: <strong>{drill.shots}</strong></span>
                <span>Make %: <strong style={{ color: 'var(--spring)' }}>{drill.makePct}</strong></span>
                <span>Best: <strong style={{ color: 'var(--volt)' }}>{drill.bestRun}</strong></span>
              </div>
              <button
                className="rb-btn active"
                onClick={() => onLaunchDrill(drill)}
              >
                ▶ Start Drill
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
