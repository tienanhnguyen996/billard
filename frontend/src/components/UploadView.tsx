import React, { useState, useRef, useEffect } from 'react';
import { UploadCloud, CheckCircle2, AlertCircle, Play, FileVideo, Cpu, Layers, Target, Compass, Sparkles } from 'lucide-react';
import { SessionData } from '../types';

interface UploadViewProps {
  onSessionReady: (sessionData: SessionData) => void;
}

interface PipelineStep {
  id: string;
  label: string;
  description: string;
}

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'ingest', label: '1. Video Ingestion', description: 'Validate codec, aspect ratio & timestamps' },
  { id: 'homography', label: '2. Table Homography', description: '4-corner cushion warp to 1000x500 radar' },
  { id: 'detection', label: '3. YOLO26/11 Ball Detection', description: 'Dual-stage candidate search & neural classifier' },
  { id: 'physics', label: '4. Trajectory & Pocket Physics', description: 'Motion segmentation, cue speeds & collisions' },
  { id: 'rating', label: '5. Shot Rating Engine', description: 'Difficulty rating, cut angles & 3-ball runouts' },
];

export const UploadView: React.FC<UploadViewProps> = ({ onSessionReady }) => {
  const [file, setFile] = useState<File | null>(null);
  const [existingVideo, setExistingVideo] = useState('SAMPLE.mp4');
  const [useExisting, setUseExisting] = useState(true);
  const [preset, setPreset] = useState<'ban_11_12' | 'matchroom' | 'topcam' | 'custom'>('ban_11_12');
  const [customCorners, setCustomCorners] = useState('558,416,1234,318,1242,672,442,716');
  const [maxSec, setMaxSec] = useState<number | ''>(60);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'processing' | 'completed' | 'failed'>('idle');
  const [progress, setProgress] = useState(0);
  const [stageName, setStageName] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [completedResult, setCompletedResult] = useState<any>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const f = e.dataTransfer.files[0];
      setFile(f);
      setUseExisting(false);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUseExisting(false);
    }
  };

  // Poll job status
  useEffect(() => {
    if (!jobId || status !== 'processing') return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/job_status?id=${jobId}`);
        const data = await res.json();
        if (data.success && data.job) {
          const j = data.job;
          setProgress(j.progress || 0);
          if (j.stage) setStageName(j.stage);

          if (j.status === 'completed') {
            setStatus('completed');
            setProgress(100);
            setCompletedResult(j.result);
            clearInterval(interval);
          } else if (j.status === 'failed') {
            setStatus('failed');
            setErrorMsg(j.error || 'Video analysis encountered an error');
            clearInterval(interval);
          }
        }
      } catch (err) {
        console.warn('Job poll error:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [jobId, status]);

  const handleStartAnalysis = async () => {
    setStatus('processing');
    setProgress(5);
    setStageName('Submitting video analysis job...');
    setErrorMsg('');
    setCompletedResult(null);

    let cornersStr = customCorners;
    if (preset === 'ban_11_12') cornersStr = '558,416,1234,318,1242,672,442,716';
    else if (preset === 'matchroom') cornersStr = '438,212,838,212,908,596,368,596';
    else if (preset === 'topcam') cornersStr = '0,0,1280,0,1280,720,0,720';

    try {
      if (useExisting) {
        // Fast path: existing video on server
        const payload = {
          video_path: existingVideo,
          preset,
          corners: cornersStr,
          end_sec: maxSec ? Number(maxSec) : null,
          title: `Upload Session (${existingVideo})`
        };

        const res = await fetch('/api/upload_video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (data.success && data.job_id) {
          setJobId(data.job_id);
          setProgress(15);
          setStageName('Video loaded. Starting CV pipeline...');
        } else {
          setStatus('failed');
          setErrorMsg(data.error || 'Failed to start analysis');
        }
      } else if (file) {
        // Direct file upload
        setStatus('uploading');
        setStageName(`Encoding ${file.name} for pipeline...`);

        // Read file as base64
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            const b64 = (reader.result as string).split(',')[1];
            setStatus('processing');
            setProgress(15);
            setStageName('Uploading video to Railbird engine...');

            const payload = {
              filename: file.name,
              video_data_base64: b64,
              preset,
              corners: cornersStr,
              end_sec: maxSec ? Number(maxSec) : null,
              title: `Uploaded Video: ${file.name}`
            };

            const res = await fetch('/api/upload_video', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success && data.job_id) {
              setJobId(data.job_id);
              setProgress(20);
              setStageName('Video uploaded. Initializing CV workers...');
            } else {
              setStatus('failed');
              setErrorMsg(data.error || 'Failed to start analysis');
            }
          } catch (e: any) {
            setStatus('failed');
            setErrorMsg(e.message || 'Error processing video upload');
          }
        };
        reader.onerror = () => {
          setStatus('failed');
          setErrorMsg('Failed to read file from disk');
        };
        reader.readAsDataURL(file);
      }
    } catch (err: any) {
      setStatus('failed');
      setErrorMsg(err.message || 'Network error communicating with Railbird server');
    }
  };

  const handleOpenInStudio = () => {
    if (completedResult && completedResult.payload) {
      onSessionReady(completedResult.payload);
    }
  };

  return (
    <div style={{ flex: 1, padding: '2rem 3rem', overflowY: 'auto', background: 'var(--ink)' }}>
      {/* Page Header */}
      <div style={{ marginBottom: '2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
          <div style={{ width: 36, height: 36, borderRadius: 8, background: 'var(--panel-alt)', border: '1px solid var(--panel-border)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--cyan)' }}>
            <UploadCloud size={20} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.6rem', fontWeight: 700, color: 'var(--bone)', margin: 0, letterSpacing: '-0.02em' }}>
              Universal Billiard Video Ingestion
            </h1>
            <p style={{ color: 'var(--mut)', fontSize: '0.88rem', margin: 0 }}>
              Process any overhead, broadcast, or billiard club CCTV camera video to auto-extract ball trajectories, shot ratings, and optimal runout lines.
            </p>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr', gap: '2rem' }}>
        {/* Left Column: File Ingestion & Calibration Preset */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Video Selection Card */}
          <div style={{ background: 'var(--panel)', border: '1px solid var(--panel-border)', borderRadius: 12, padding: '1.5rem' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--bone)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileVideo size={18} color="var(--volt)" /> 1. Select Source Video
            </h2>

            {/* Ingestion Mode Toggle */}
            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
              <button
                className={`rb-btn ${useExisting ? 'active' : ''}`}
                onClick={() => setUseExisting(true)}
                style={{ flex: 1 }}
              >
                ⚡ Fast-Load Server Video
              </button>
              <button
                className={`rb-btn ${!useExisting ? 'active' : ''}`}
                onClick={() => setUseExisting(false)}
                style={{ flex: 1 }}
              >
                📁 Upload Local Video File
              </button>
            </div>

            {useExisting ? (
              <div>
                <label style={{ display: 'block', fontSize: '0.82rem', color: 'var(--mut)', marginBottom: '0.4rem' }}>
                  Available Server Videos:
                </label>
                <select
                  value={existingVideo}
                  onChange={(e) => setExistingVideo(e.target.value)}
                  style={{
                    width: '100%',
                    padding: '0.65rem 0.9rem',
                    background: 'var(--ink-2)',
                    border: '1px solid var(--panel-border)',
                    borderRadius: 8,
                    color: 'var(--bone)',
                    fontSize: '0.88rem'
                  }}
                >
                  <option value="SAMPLE.mp4">SAMPLE.mp4 (Billiard Club CCTV 9-Ball)</option>
                  <option value="hoang_sao.mp4">hoang_sao.mp4 (European Open Broadcast)</option>
                </select>
              </div>
            ) : (
              <div>
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="video/mp4,video/quicktime,video/x-matroska,video/avi"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
                <div
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleFileDrop}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: `2px dashed ${isDragging ? 'var(--cyan)' : 'rgba(255, 255, 255, 0.14)'}`,
                    borderRadius: 10,
                    padding: '2.5rem 1.5rem',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: isDragging ? 'rgba(70, 230, 255, 0.05)' : 'var(--ink-2)',
                    transition: 'border-color 0.2s, background 0.2s'
                  }}
                >
                  <UploadCloud size={36} color={file ? 'var(--spring)' : 'var(--mut)'} style={{ margin: '0 auto 0.75rem' }} />
                  {file ? (
                    <div>
                      <div style={{ color: 'var(--spring)', fontWeight: 600, fontSize: '0.95rem' }}>{file.name}</div>
                      <div style={{ color: 'var(--mut)', fontSize: '0.78rem', marginTop: 4 }}>
                        {(file.size / (1024 * 1024)).toFixed(1)} MB • Click to replace
                      </div>
                    </div>
                  ) : (
                    <div>
                      <div style={{ color: 'var(--bone)', fontWeight: 500, fontSize: '0.92rem' }}>
                        Drag & Drop billiard video or <span style={{ color: 'var(--cyan)', textDecoration: 'underline' }}>browse</span>
                      </div>
                      <div style={{ color: 'var(--mut)', fontSize: '0.78rem', marginTop: 4 }}>
                        Supports MP4, MOV, MKV up to 4K resolution
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Camera & Table Calibration Preset */}
          <div style={{ background: 'var(--panel)', border: '1px solid var(--panel-border)', borderRadius: 12, padding: '1.5rem' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--bone)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Compass size={18} color="var(--cyan)" /> 2. Camera Angle & Cushion Calibration
            </h2>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1.2rem' }}>
              <button
                className={`rb-btn ${preset === 'ban_11_12' ? 'active' : ''}`}
                onClick={() => setPreset('ban_11_12')}
                style={{ flexDirection: 'column', height: 'auto', padding: '0.75rem 0.5rem', textAlign: 'center' }}
              >
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>Club CCTV</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--mut)', marginTop: 2 }}>Angled Club Table (9ft)</div>
              </button>

              <button
                className={`rb-btn ${preset === 'matchroom' ? 'active' : ''}`}
                onClick={() => setPreset('matchroom')}
                style={{ flexDirection: 'column', height: 'auto', padding: '0.75rem 0.5rem', textAlign: 'center' }}
              >
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>TV Broadcast</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--mut)', marginTop: 2 }}>Matchroom Diamond</div>
              </button>

              <button
                className={`rb-btn ${preset === 'topcam' ? 'active' : ''}`}
                onClick={() => setPreset('topcam')}
                style={{ flexDirection: 'column', height: 'auto', padding: '0.75rem 0.5rem', textAlign: 'center' }}
              >
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>Overhead Top-Cam</div>
                <div style={{ fontSize: '0.72rem', color: 'var(--mut)', marginTop: 2 }}>1:1 Bird's Eye View</div>
              </button>
            </div>

            {/* Max Duration Slider */}
            <div style={{ marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.4rem' }}>
                <span style={{ color: 'var(--mut)' }}>Process Window Duration:</span>
                <span style={{ color: 'var(--bone)', fontWeight: 600 }}>{maxSec ? `${maxSec} seconds` : 'Full Video'}</span>
              </div>
              <input
                type="range"
                min="15"
                max="180"
                step="15"
                value={maxSec || 60}
                onChange={(e) => setMaxSec(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--volt)' }}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--mut)', marginTop: 2 }}>
                <span>15s (Quick Test)</span>
                <span>60s</span>
                <span>120s</span>
                <span>180s (Full Match)</span>
              </div>
            </div>

            {/* Launch Button */}
            <button
              className="rb-btn rb-btn-primary"
              onClick={handleStartAnalysis}
              disabled={status === 'processing' || status === 'uploading' || (!useExisting && !file)}
              style={{
                width: '100%',
                padding: '0.85rem 1rem',
                fontSize: '0.95rem',
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.6rem',
                background: status === 'processing' ? 'var(--mut)' : 'linear-gradient(135deg, var(--volt), #ffd500)',
                color: 'var(--ink)',
                cursor: status === 'processing' ? 'not-allowed' : 'pointer'
              }}
            >
              <Sparkles size={18} />
              {status === 'processing' ? 'Processing Video...' : status === 'uploading' ? 'Uploading...' : 'Launch Universal CV Pipeline'}
            </button>
          </div>
        </div>

        {/* Right Column: Real-time Pipeline Progress & Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ background: 'var(--panel)', border: '1px solid var(--panel-border)', borderRadius: 12, padding: '1.5rem', flex: 1, display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--bone)', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={18} color="var(--spring)" /> AI Video Processing Pipeline
            </h2>

            {/* Progress Bar */}
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.84rem', marginBottom: '0.5rem' }}>
                <span style={{ color: 'var(--bone)', fontWeight: 600 }}>{stageName || 'Pipeline Ready'}</span>
                <span style={{ color: 'var(--volt)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{Math.round(progress)}%</span>
              </div>
              <div style={{ width: '100%', height: 8, background: 'var(--ink-2)', borderRadius: 4, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${progress}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, var(--cyan), var(--volt), var(--spring))',
                    transition: 'width 0.4s ease-out'
                  }}
                />
              </div>
            </div>

            {/* Pipeline Stepper */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', flex: 1 }}>
              {PIPELINE_STEPS.map((step, idx) => {
                const stepThreshold = (idx + 1) * 20;
                const isDone = progress >= stepThreshold || status === 'completed';
                const isCurrent = progress < stepThreshold && progress >= idx * 20 && status === 'processing';

                return (
                  <div
                    key={step.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.9rem',
                      padding: '0.75rem 0.9rem',
                      borderRadius: 8,
                      background: isCurrent ? 'rgba(70, 230, 255, 0.08)' : 'var(--ink-2)',
                      border: `1px solid ${isCurrent ? 'var(--cyan)' : 'var(--panel-border)'}`
                    }}
                  >
                    <div style={{ color: isDone ? 'var(--spring)' : isCurrent ? 'var(--cyan)' : 'var(--mut)' }}>
                      {isDone ? <CheckCircle2 size={18} /> : <Layers size={18} />}
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: isDone || isCurrent ? 'var(--bone)' : 'var(--mut)' }}>
                        {step.label}
                      </div>
                      <div style={{ fontSize: '0.74rem', color: 'var(--mut)' }}>
                        {step.description}
                      </div>
                    </div>
                    {isCurrent && (
                      <span style={{ fontSize: '0.7rem', color: 'var(--cyan)', fontWeight: 700, padding: '2px 8px', borderRadius: 4, background: 'rgba(70, 230, 255, 0.15)' }}>
                        ACTIVE
                      </span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Error Message */}
            {status === 'failed' && (
              <div style={{ marginTop: '1rem', padding: '0.85rem 1rem', borderRadius: 8, background: 'rgba(214, 69, 69, 0.15)', border: '1px solid var(--red)', display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--bone)', fontSize: '0.85rem' }}>
                <AlertCircle size={18} color="var(--red)" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Completed Result Card */}
            {status === 'completed' && completedResult && (
              <div style={{ marginTop: '1.25rem', padding: '1.25rem', borderRadius: 10, background: 'rgba(91, 255, 158, 0.08)', border: '1px solid var(--spring)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--spring)', fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.75rem' }}>
                  <CheckCircle2 size={20} /> Analysis Complete & Session Ready!
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
                  <div style={{ background: 'var(--ink-2)', padding: '0.6rem 0.75rem', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--mut)' }}>SHOTS DETECTED</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--volt)', fontFamily: 'var(--font-mono)' }}>
                      {completedResult.shots_count || 0}
                    </div>
                  </div>
                  <div style={{ background: 'var(--ink-2)', padding: '0.6rem 0.75rem', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--mut)' }}>POTTING ACCURACY</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--spring)', fontFamily: 'var(--font-mono)' }}>
                      {completedResult.pot_pct || 0}%
                    </div>
                  </div>
                  <div style={{ background: 'var(--ink-2)', padding: '0.6rem 0.75rem', borderRadius: 6 }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--mut)' }}>DURATION</div>
                    <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>
                      {completedResult.duration_sec ? `${Math.round(completedResult.duration_sec)}s` : 'N/A'}
                    </div>
                  </div>
                </div>

                <button
                  className="rb-btn rb-btn-primary"
                  onClick={handleOpenInStudio}
                  style={{
                    width: '100%',
                    padding: '0.8rem 1rem',
                    fontSize: '0.92rem',
                    fontWeight: 700,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                    background: 'var(--spring)',
                    color: 'var(--ink)'
                  }}
                >
                  <Play size={16} fill="var(--ink)" /> Open Session in Player Studio
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
