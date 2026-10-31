export interface Ball {
  label: string;
  x: number;
  y: number;
  rx: number;
  ry: number;
  color?: number[] | string;
}

export interface TimelineSample {
  time: number;
  frame: number;
  balls: Ball[];
  cue_speed_kmh?: number;
}

export interface AimLine {
  cb_x: number;
  cb_y: number;
  ob_x?: number;
  ob_y?: number;
  gb_x: number;
  gb_y: number;
  pock_x: number;
  pock_y: number;
  cb_rx: number;
  cb_ry: number;
  ob_rx?: number;
  ob_ry?: number;
  gb_rx: number;
  gb_ry: number;
  pock_rx: number;
  pock_ry: number;
}

export interface Shot {
  shot_number: number;
  shot_type?: string;
  player?: string;
  target_ball?: string;
  outcome: string;
  grade?: string;
  cut_angle_deg?: number;
  margin_deg?: number;
  difficulty_score?: number;
  max_speed_kmh?: number;
  speed_gear?: string;
  start_time: number;
  end_time: number;
  stroke_time?: number;
  impact_time?: number;
  stop_time?: number;
  is_highlight?: boolean;
  coach_tip?: string;
  pocket?: string;
  cue_tip?: {
    tx: number;
    ty: number;
    spin_name: string;
  };
  aim_line?: AimLine;
  cue_path?: Array<{ t: number; x: number; y: number; rx: number; ry: number; speed_kmh?: number }>;
  obj_path?: Array<{ t: number; x: number; y: number; rx: number; ry: number; speed_kmh?: number }>;
  break_paths?: Array<{
    label: string;
    color: number[] | string;
    path: Array<{ t: number; x: number; y: number; rx: number; ry: number }>;
  }>;
}

export interface SessionData {
  session_id: string;
  match_title?: string;
  player_1?: string;
  video_file: string;
  duration_sec: number;
  fps: number;
  table_poly: number[][];
  pockets: Array<{ id: string; name: string; x: number; y: number; r: number; rx: number; ry: number }>;
  summary?: {
    tpa?: number;
    pot_pct?: number;
    total_shots?: number;
  };
  shots: Shot[];
  timeline_sample: TimelineSample[];
}
