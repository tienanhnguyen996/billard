import React, { useState, useEffect, useRef } from 'react';
import { SessionData, Shot, Ball } from '../types';
import { PlayerAiStatsRail } from './PlayerAiStatsRail';
import { ShotTimeline } from './ShotTimeline';
import { TableRadar } from './TableRadar';
import { ShotsPlaylist } from './ShotsPlaylist';
import { RunTrackingPip } from './RunTrackingPip';
import { drawVideoOverlayShotline, getBallShortLabel, bgrToCss, getShotTiming } from '../utils/geometry';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  Gauge,
  Layers,
  Maximize2
} from 'lucide-react';

interface PlayerStudioProps {
  session: SessionData;
  sessionMode: string;
  onSelectSession: (mode: string) => void;
  onPracticeInLab: (shot: Shot) => void;
}

export const PlayerStudio: React.FC<PlayerStudioProps> = ({
  session,
  sessionMode,
  onSelectSession,
  onPracticeInLab
}) => {
  const [activeShotIndex, setActiveShotIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [shotFilter, setShotFilter] = useState('all');
  const [hudMode, setHudMode] = useState<'full' | 'rings' | 'off'>('full');
  const [autoSkipDowntime, setAutoSkipDowntime] = useState(false);
  const [isPipMode, setIsPipMode] = useState(false);
  const [currentBalls, setCurrentBalls] = useState<Ball[]>([]);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  const shots = session.shots || [];
  const currentShot = shots[activeShotIndex] || shots[0];

  // 60 FPS linear interpolation
  const getInterpolatedSample = (t: number) => {
    const samples = session.timeline_sample || [];
    if (!samples.length) return null;
    if (t <= samples[0].time) return samples[0];
    if (t >= samples[samples.length - 1].time) return samples[samples.length - 1];

    let low = 0, high = samples.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (samples[mid].time < t) low = mid + 1;
      else high = mid - 1;
    }
    const s0 = samples[Math.max(0, low - 1)];
    const s1 = samples[Math.min(samples.length - 1, low)];
    if (s0 === s1) return s0;

    const alpha = (t - s0.time) / (s1.time - s0.time || 1e-4);
    const ballsMap: Record<string, { b0?: Ball; b1?: Ball }> = {};
    (s0.balls || []).forEach(b => { ballsMap[b.label] = { b0: b }; });
    (s1.balls || []).forEach(b => {
      if (!ballsMap[b.label]) ballsMap[b.label] = {};
      ballsMap[b.label].b1 = b;
    });

    const interp: Ball[] = [];
    for (const label in ballsMap) {
      const { b0, b1 } = ballsMap[label];
      if (b0 && b1) {
        interp.push({
          label,
          x: b0.x + alpha * (b1.x - b0.x),
          y: b0.y + alpha * (b1.y - b0.y),
          rx: b0.rx + alpha * (b1.rx - b0.rx),
          ry: b0.ry + alpha * (b1.ry - b0.ry),
          color: b1.color || b0.color
        });
      } else if (b0) {
        interp.push(b0);
      } else if (b1) {
        interp.push(b1);
      }
    }
    return { time: t, frame: Math.round(s0.frame + alpha * (s1.frame - s0.frame)), balls: interp };
  };

  // Video playback loop
  useEffect(() => {
    let animId: number;

    const loop = () => {
      const v = videoRef.current;
      const overlay = overlayRef.current;
      if (!v || !overlay) {
        animId = requestAnimationFrame(loop);
        return;
      }

      const t = v.currentTime;
      setCurrentTime(t);

      // Lock canvas dimensions to video
      if (v.videoWidth && (overlay.width !== v.videoWidth || overlay.height !== v.videoHeight)) {
        overlay.width = v.videoWidth;
        overlay.height = v.videoHeight;
      }

      const ctx = overlay.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        const sample = getInterpolatedSample(t);
        const balls = sample ? sample.balls : [];
        setCurrentBalls(balls);

        // Synchronize active shot with playback time
        const liveIdx = shots.findIndex(s => {
          const { strokeT, stopT } = getShotTiming(s);
          return t >= (strokeT - 1.8) && t <= (stopT + 0.6);
        });
        if (liveIdx >= 0 && !v.paused && liveIdx !== activeShotIndex) {
          setActiveShotIndex(liveIdx);
        }

        // Render HUD
        if (hudMode !== 'off') {
          // 1. Draw Shotline & Trajectories (Mathematically exact!)
          if (hudMode === 'full' && currentShot) {
            drawVideoOverlayShotline(ctx, currentShot, balls, t, v.paused);
          }

          // 2. Draw Crisp Hollow Rings around balls
          balls.forEach(b => {
            const isCue = b.label.toLowerCase().includes('cue');
            const tag = getBallShortLabel(b.label);
            const strokeColor = isCue ? '#ffffff' : bgrToCss(b.color, 0.95);

            ctx.save();
            ctx.beginPath();
            ctx.arc(b.x, b.y, isCue ? 11 : 9.5, 0, Math.PI * 2);
            ctx.lineWidth = 2.2;
            ctx.strokeStyle = strokeColor;
            ctx.stroke();

            // Label pill above ball
            const labelY = b.y - 14;
            ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            ctx.beginPath();
            ctx.rect(b.x - 9, labelY - 6, 18, 12);
            ctx.fill();
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 9px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(tag, b.x, labelY);
            ctx.restore();
          });
        }
      }

      // Auto-skip downtime
      if (autoSkipDowntime && !v.paused) {
        let inShot = false;
        for (let i = 0; i < shots.length; i++) {
          const { strokeT, stopT } = getShotTiming(shots[i]);
          if (t >= strokeT - 1.8 && t <= stopT + 0.6) {
            inShot = true;
            break;
          }
        }
        if (!inShot) {
          const next = shots.find(s => {
            const { strokeT } = getShotTiming(s);
            return strokeT - 1.8 > t + 0.3;
          });
          if (next) {
            const { strokeT } = getShotTiming(next);
            v.currentTime = Math.max(0, strokeT - 1.8);
          }
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [shots, activeShotIndex, currentShot, hudMode, autoSkipDowntime]);

  // Stepping & Controls
  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play();
      setIsPlaying(true);
    } else {
      v.pause();
      setIsPlaying(false);
    }
  };

  const stepFrame = (delta: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.pause();
    setIsPlaying(false);
    v.currentTime = Math.max(0, v.currentTime + delta * (1.0 / (session.fps || 30)));
  };

  const seekShot = (index: number) => {
    setActiveShotIndex(index);
    const targetShot = shots[index];
    if (targetShot && videoRef.current) {
      const { strokeT } = getShotTiming(targetShot);
      videoRef.current.currentTime = Math.max(0, strokeT - 1.8);
    }
  };

  const setSpeed = (spd: number) => {
    setPlaybackSpeed(spd);
    if (videoRef.current) videoRef.current.playbackRate = spd;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
      {/* Top App Sub-Bar */}
      <div className="rb-topbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--mut)' }}>SESSION:</span>
          <div className="rb-session-pills">
            <button
              className={`rb-pill-btn ${sessionMode === 'topcam' ? 'active' : ''}`}
              onClick={() => onSelectSession('topcam')}
            >
              🎯 Top-Cam Drill (1:1 Overhead)
            </button>
            <button
              className={`rb-pill-btn ${sessionMode === 'hs' ? 'active' : ''}`}
              onClick={() => onSelectSession('hs')}
            >
              🏆 Dương Quốc Hoàng (Broadcast)
            </button>
            <button
              className={`rb-pill-btn ${sessionMode === 'club' ? 'active' : ''}`}
              onClick={() => onSelectSession('club')}
            >
              🎱 Club CCTV: Railbird 9-Ball
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            className={`rb-btn ${shotFilter === 'best' ? 'active' : ''}`}
            onClick={() => setShotFilter(shotFilter === 'best' ? 'all' : 'best')}
          >
            ⭐ Top Highlights Only
          </button>
          <button
            className={`rb-btn ${isPipMode ? 'active' : ''}`}
            onClick={() => setIsPipMode(!isPipMode)}
          >
            <Maximize2 size={14} />
            <span>Run-Tracking PiP</span>
          </button>
        </div>
      </div>

      {/* Main Studio 2-Column Grid */}
      <div className="rb-studio-grid">
        {/* Left Column: Player Main Row + Timeline + Controls Bar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', minWidth: 0 }}>
          <div className="rb-card">
            {isPipMode ? (
              <RunTrackingPip
                videoRef={videoRef}
                shots={shots}
                activeShotIndex={activeShotIndex}
                balls={currentBalls}
                currentTime={currentTime}
                onClose={() => setIsPipMode(false)}
              />
            ) : (
              <div className="rb-player-row">
                <PlayerAiStatsRail
                  shot={currentShot}
                  playerName={currentShot?.player || (sessionMode === 'hs' ? 'Dương Quốc Hoàng' : 'Club Player')}
                  playerRating={sessionMode === 'hs' ? '98.5' : '74.2'}
                />

                <div className="rb-stage-wrap">
                  <video
                    ref={videoRef}
                    playsInline
                    preload="auto"
                    src={session.video_file || 'hoang_sao.mp4'}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                  />
                  <canvas ref={overlayRef} />
                </div>
              </div>
            )}

            {/* Proportional Shot Timeline Scrubber */}
            <ShotTimeline
              shots={shots}
              activeShotIndex={activeShotIndex}
              duration={session.duration_sec || 60}
              currentTime={currentTime}
              onSeek={(t) => {
                if (videoRef.current) videoRef.current.currentTime = t;
              }}
              onSelectShot={seekShot}
            />

            {/* Controls Bar */}
            <div className="rb-controls-bar">
              <div className="rb-btn-group">
                <button className="rb-btn" onClick={() => stepFrame(-1)} title="Step -1 Frame (Shortcut: ,)">
                  <ChevronLeft size={14} /> -1F
                </button>
                <button className="rb-btn active" onClick={togglePlay} title="Play / Pause (Space)">
                  {isPlaying ? <Pause size={14} /> : <Play size={14} />}
                  <span>{isPlaying ? 'Pause' : 'Play'}</span>
                </button>
                <button className="rb-btn" onClick={() => stepFrame(1)} title="Step +1 Frame (Shortcut: .)">
                  +1F <ChevronRight size={14} />
                </button>
                <button
                  className="rb-btn"
                  onClick={() => seekShot(Math.max(0, activeShotIndex - 1))}
                  title="Previous Shot ([)"
                >
                  <SkipBack size={14} /> Prev
                </button>
                <button
                  className="rb-btn"
                  onClick={() => seekShot(Math.min(shots.length - 1, activeShotIndex + 1))}
                  title="Next Shot (])"
                >
                  Next <SkipForward size={14} />
                </button>
              </div>

              {/* Speed Buttons */}
              <div className="rb-btn-group">
                <span style={{ fontSize: '0.72rem', color: 'var(--mut)', marginRight: 2 }}>SPEED:</span>
                {[0.25, 0.5, 1.0, 1.5, 2.0].map(s => (
                  <button
                    key={s}
                    className={`rb-btn ${playbackSpeed === s ? 'active' : ''}`}
                    onClick={() => setSpeed(s)}
                    style={{ padding: '0.35rem 0.55rem' }}
                  >
                    {s}x
                  </button>
                ))}
              </div>

              {/* HUD Mode & Auto-skip */}
              <div className="rb-btn-group">
                <button
                  className="rb-btn"
                  onClick={() => setHudMode(hudMode === 'full' ? 'rings' : hudMode === 'rings' ? 'off' : 'full')}
                >
                  <Layers size={14} />
                  <span>HUD: {hudMode.toUpperCase()}</span>
                </button>
                <button
                  className={`rb-btn ${autoSkipDowntime ? 'active' : ''}`}
                  onClick={() => setAutoSkipDowntime(!autoSkipDowntime)}
                >
                  <Gauge size={14} />
                  <span>Auto-Skip Downtime</span>
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: 2D Table Radar & Shots List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', minWidth: 0 }}>
          <div className="rb-card" style={{ padding: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.65rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--bone)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                🎯 2D Shot Diagram
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--spring)' }}>
                60 FPS Interp
              </span>
            </div>
            <TableRadar
              shot={currentShot}
              balls={currentBalls}
              sessionMode={sessionMode}
              currentTime={currentTime}
            />
          </div>

          <div className="rb-card" style={{ padding: '0.85rem' }}>
            <ShotsPlaylist
              shots={shots}
              activeShotIndex={activeShotIndex}
              shotFilter={shotFilter}
              onFilterChange={setShotFilter}
              onSelectShot={seekShot}
              onPracticeInLab={onPracticeInLab}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
