import React, { useState } from 'react';

const SENSOR_CONFIG = {
  slope_angle: { label: 'Slope Angle', min: 10, max: 85, step: 1 },
  slope_height: { label: 'Slope Height', min: 10, max: 120, step: 1 },
  cohesion: { label: 'Cohesion', min: 0, max: 40, step: 0.5 },
  friction_angle: { label: 'Friction Angle', min: 5, max: 45, step: 0.5 },
  unit_weight: { label: 'Unit Weight', min: 12, max: 26, step: 0.1 },
  pore_water_pressure_ratio: { label: 'Pore Water Pressure Ratio', min: 0, max: 1, step: 0.01 },
  reinforcement_numeric: { label: 'Reinforcement Numeric', min: 0, max: 1, step: 1 },
};

function App() {
  // === UI STATE ===
  const [activeTab, setActiveTab] = useState('telemetry'); // 'telemetry' or 'vision'

  // === TELEMETRY STATE ===
  const [sensors, setSensors] = useState({
    slope_angle: 55.0,
    slope_height: 65.0,
    cohesion: 12.0,
    friction_angle: 26.0,
    unit_weight: 19.5,
    pore_water_pressure_ratio: 0.35,
    reinforcement_numeric: 0,
  });
  const [reinforcementType, setReinforcementType] = useState('none');
  const [riskScore, setRiskScore] = useState(null);
  const [status, setStatus] = useState("AWAITING TELEMETRY");
  const [recommendation, setRecommendation] = useState("");
  const [isTelLoading, setIsTelLoading] = useState(false);
  const [telError, setTelError] = useState(null);
  const [lastTelemetryPayload, setLastTelemetryPayload] = useState(null);
  const [lastTelemetryResponse, setLastTelemetryResponse] = useState(null);

  // === VISION STATE ===
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [hazards, setHazards] = useState([]);
  const [isVisLoading, setIsVisLoading] = useState(false);
  const [visError, setVisError] = useState(null);

  // === TELEMETRY FUNCTIONS ===
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setSensors({ ...sensors, [name]: parseFloat(value) });
  };

  const runTelemetry = async () => {
    setIsTelLoading(true);
    setTelError(null);
    setRecommendation("");
    try {
      const telemetryPayload = {
        ...sensors,
        reinforcement_type_geosynthetics: reinforcementType === 'geosynthetics' ? 1 : 0,
        reinforcement_type_retaining_wall: reinforcementType === 'retaining_wall' ? 1 : 0,
        reinforcement_type_soil_nailing: reinforcementType === 'soil_nailing' ? 1 : 0,
      };
      setLastTelemetryPayload(telemetryPayload);

      const response = await fetch('http://127.0.0.1:8000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ telemetry: telemetryPayload }) 
      });
      if (!response.ok) throw new Error("Failed to connect to AI Core.");
      const data = await response.json();
      setRiskScore(data.risk_score);
      setStatus(data.status);
      setLastTelemetryResponse(data);
      if (data.recommendation) setRecommendation(data.recommendation);
    } catch (err) {
      setTelError(err.message);
    } finally {
      setIsTelLoading(false);
    }
  };

  const applyScenario = (mode) => {
    if (mode === 'safe') {
      setSensors({
        slope_angle: 25.0,
        slope_height: 25.0,
        cohesion: 30.0,
        friction_angle: 40.0,
        unit_weight: 15.5,
        pore_water_pressure_ratio: 0.1,
        reinforcement_numeric: 1,
      });
      setReinforcementType('retaining_wall');
      return;
    }

    setSensors({
      slope_angle: 82.0,
      slope_height: 95.0,
      cohesion: 5.0,
      friction_angle: 12.0,
      unit_weight: 24.5,
      pore_water_pressure_ratio: 0.95,
      reinforcement_numeric: 0,
    });
    setReinforcementType('none');
  };

  const getStatusColor = () => {
    if (riskScore > 75) return "text-rose-600 border-rose-200 bg-rose-50 shadow-[0_0_40px_rgba(225,29,72,0.15)]";
    if (riskScore > 30) return "text-amber-600 border-amber-200 bg-amber-50 shadow-[0_0_40px_rgba(217,119,6,0.15)]";
    if (riskScore !== null) return "text-emerald-600 border-emerald-200 bg-emerald-50 shadow-[0_0_40px_rgba(5,150,105,0.15)]";
    return "text-slate-400 border-slate-200 bg-white shadow-xl";
  };

  // === VISION FUNCTIONS ===
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setHazards([]); // clear old boxes
    }
  };

  const runVision = async () => {
    if (!selectedImage) return;
    setIsVisLoading(true);
    setVisError(null);
    setHazards([]);

    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/vision', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error("Vision AI is offline or training.");
      const data = await response.json();
      setHazards(data.data || []);
    } catch (err) {
      setVisError(err.message);
    } finally {
      setIsVisLoading(false);
    }
  };

  // Capture original image dimensions to draw boxes accurately
  const onImageLoad = ({ target: img }) => {
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 animate-gradient-x text-slate-800 p-8 font-sans transition-all duration-700 relative overflow-hidden">
      
      <div className="absolute top-[-10%] left-[-10%] w-[500px] h-[500px] bg-blue-200/50 rounded-full blur-[100px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[500px] h-[500px] bg-indigo-200/50 rounded-full blur-[100px] pointer-events-none"></div>

      {/* HEADER & NAVIGATION */}
      <header className="mb-8 relative z-10 flex flex-col md:flex-row justify-between items-center gap-6 border-b border-slate-200 pb-6">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-indigo-700 drop-shadow-sm">
            ⛰️ Project Lithos
          </h1>
          <p className="text-slate-500 mt-2 tracking-wide font-medium">Enterprise Geo-Spatial AI Command Center</p>
        </div>
        
        {/* Tab Switcher */}
        <div className="flex bg-white/50 backdrop-blur-md p-1.5 rounded-2xl shadow-sm border border-slate-200/50">
          <button 
            onClick={() => setActiveTab('telemetry')}
            className={`px-6 py-2.5 rounded-xl font-bold transition-all duration-300 ${activeTab === 'telemetry' ? 'bg-white shadow-md text-blue-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            📡 Sensor Telemetry
          </button>
          <button 
            onClick={() => setActiveTab('vision')}
            className={`px-6 py-2.5 rounded-xl font-bold transition-all duration-300 ${activeTab === 'vision' ? 'bg-white shadow-md text-indigo-600' : 'text-slate-500 hover:text-slate-700'}`}
          >
            🚁 Drone Vision
          </button>
        </div>
      </header>

      {/* ========================================== */}
      {/* TAB 1: SENSOR TELEMETRY                    */}
      {/* ========================================== */}
      {activeTab === 'telemetry' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10 animate-fade-in-up">
          <div className="bg-white/80 backdrop-blur-xl border border-white p-8 rounded-3xl shadow-xl transition-all duration-500">
            <h2 className="text-xl font-bold mb-8 text-slate-700 flex items-center gap-2">
              <span className="animate-pulse text-blue-500">📡</span> Sensor Calibration
            </h2>
            {Object.keys(sensors).map((key) => (
              <div key={key} className="mb-6 group">
                <label className="block text-sm font-semibold text-slate-600 mb-3 capitalize group-hover:text-blue-600 transition-colors">
                  {(SENSOR_CONFIG[key] && SENSOR_CONFIG[key].label) || key.replace(/_/g, ' ')}
                </label>
                <input
                  type="range"
                  name={key}
                  min={SENSOR_CONFIG[key] ? SENSOR_CONFIG[key].min : 0}
                  max={SENSOR_CONFIG[key] ? SENSOR_CONFIG[key].max : 100}
                  step={SENSOR_CONFIG[key] ? SENSOR_CONFIG[key].step : 0.5}
                  value={sensors[key]}
                  onChange={handleInputChange}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-blue-600 focus:outline-none focus:ring-4 focus:ring-blue-500/30 transition-all"
                />
                <div className="text-right text-sm text-slate-500 mt-2 font-mono font-medium">{sensors[key]}</div>
              </div>
            ))}

            <div className="mb-6 group">
              <label className="block text-sm font-semibold text-slate-600 mb-3 group-hover:text-blue-600 transition-colors">
                Reinforcement Type
              </label>
              <select
                value={reinforcementType}
                onChange={(e) => setReinforcementType(e.target.value)}
                className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-slate-700 font-medium focus:outline-none focus:ring-4 focus:ring-blue-500/20"
              >
                <option value="none">None</option>
                <option value="geosynthetics">Geosynthetics</option>
                <option value="retaining_wall">Retaining Wall</option>
                <option value="soil_nailing">Soil Nailing</option>
              </select>
            </div>

            <p className="text-xs text-slate-500 mb-2">After changing values, click "Run Physics Diagnostics" to refresh prediction.</p>
            <div className="grid grid-cols-2 gap-2 mb-4">
              <button
                onClick={() => applyScenario('safe')}
                className="px-3 py-2 rounded-lg text-sm font-bold bg-emerald-50 border border-emerald-200 text-emerald-700 hover:bg-emerald-100 transition-colors"
              >
                Load Safe Scenario
              </button>
              <button
                onClick={() => applyScenario('critical')}
                className="px-3 py-2 rounded-lg text-sm font-bold bg-rose-50 border border-rose-200 text-rose-700 hover:bg-rose-100 transition-colors"
              >
                Load Critical Scenario
              </button>
            </div>
            <div className="relative mt-8 group">
              {!isTelLoading && <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur opacity-25 group-hover:opacity-75 transition duration-1000 animate-pulse"></div>}
              <button
                onClick={runTelemetry}
                disabled={isTelLoading}
                className="relative w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-4 px-6 rounded-2xl shadow-[0_10px_20px_rgba(79,70,229,0.3)] transition-all overflow-hidden flex justify-center items-center gap-2"
              >
                <span className="relative z-10">{isTelLoading ? "Analyzing..." : "Run Physics Diagnostics"}</span>
              </button>
            </div>
            {telError && <p className="mt-4 text-sm text-rose-600 bg-rose-50 p-4 rounded-xl border border-rose-100">{telError}</p>}

            <details className="mt-4 bg-slate-50 border border-slate-200 rounded-xl p-3">
              <summary className="cursor-pointer font-semibold text-slate-700">Telemetry Debug (payload/response)</summary>
              <div className="mt-2 text-xs text-slate-600">
                <p className="font-bold mb-1">Sent payload:</p>
                <pre className="whitespace-pre-wrap break-words bg-white border border-slate-200 rounded p-2">{JSON.stringify(lastTelemetryPayload, null, 2)}</pre>
                <p className="font-bold mt-2 mb-1">API response:</p>
                <pre className="whitespace-pre-wrap break-words bg-white border border-slate-200 rounded p-2">{JSON.stringify(lastTelemetryResponse, null, 2)}</pre>
              </div>
            </details>
          </div>

          <div className="lg:col-span-2 flex flex-col justify-center items-center bg-white/80 backdrop-blur-xl border border-white p-8 rounded-3xl shadow-xl relative overflow-hidden transition-all duration-500">
            <h2 className="text-2xl font-bold text-slate-700 mb-10 z-10 tracking-wide">Global Failure Risk</h2>
            <div className={`flex flex-col items-center justify-center w-80 h-80 rounded-full border-[8px] z-10 transition-all duration-700 transform ${riskScore > 75 ? 'scale-105' : 'scale-100'} ${getStatusColor()}`}>
              <span className="text-7xl font-extrabold tracking-tighter">{riskScore !== null ? `${riskScore}%` : "--"}</span>
              <span className="text-sm tracking-[0.3em] mt-4 font-mono font-bold opacity-70">RISK INDEX</span>
            </div>
            <div className={`mt-10 text-xl font-extrabold tracking-widest z-10 transition-all duration-700 px-8 py-3 rounded-full ${getStatusColor().split(' ')[0]} ${getStatusColor().split(' ')[2]}`}>
              STATUS: {status}
            </div>
            {recommendation && (
              <div className="mt-8 z-10 max-w-lg text-center bg-indigo-50 border border-indigo-200 p-5 rounded-2xl shadow-sm">
                <p className="text-indigo-800 font-bold leading-relaxed">{recommendation}</p>
              </div>
            )}
            {riskScore > 75 && <div className="absolute inset-0 bg-rose-500/10 animate-pulse z-0 rounded-3xl"></div>}
          </div>
        </div>
      )}

      {/* ========================================== */}
      {/* TAB 2: DRONE VISION                          */}
      {/* ========================================== */}
      {activeTab === 'vision' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 relative z-10 animate-fade-in-up">
          
          {/* Controls */}
          <div className="bg-white/80 backdrop-blur-xl border border-white p-8 rounded-3xl shadow-xl transition-all duration-500">
            <h2 className="text-xl font-bold mb-6 text-slate-700 flex items-center gap-2">
              <span className="animate-pulse text-indigo-500">🚁</span> Aerial Watchdog
            </h2>
            
            <p className="text-sm text-slate-500 mb-6 font-medium leading-relaxed">
              Upload high-resolution drone imagery. The YOLOv8 Vision AI will autonomously detect structural cracks, seepage, and rockfall debris.
            </p>

            <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-slate-300 border-dashed rounded-2xl cursor-pointer bg-slate-50 hover:bg-slate-100 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-8 h-8 mb-4 text-slate-400" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16"><path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/></svg>
                <p className="mb-2 text-sm text-slate-500 font-bold">Click to upload image</p>
                <p className="text-xs text-slate-400">JPEG, PNG up to 10MB</p>
              </div>
              <input type="file" className="hidden" accept="image/*" onChange={handleImageUpload} />
            </label>

            <button
              onClick={runVision}
              disabled={isVisLoading || !selectedImage}
              className="w-full mt-6 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-bold py-4 px-6 rounded-2xl shadow-[0_10px_20px_rgba(99,102,241,0.3)] transition-all disabled:opacity-50 flex justify-center items-center gap-2"
            >
              {isVisLoading ? "Scanning Pixels..." : "Run Computer Vision"}
            </button>

            {visError && <p className="mt-4 text-sm text-rose-600 bg-rose-50 p-4 rounded-xl border border-rose-100">{visError}</p>}

            {/* Hazard Summary List */}
            {hazards.length > 0 && (
              <div className="mt-8">
                <h3 className="text-sm font-bold text-slate-700 mb-3 uppercase tracking-wider">Detected Hazards ({hazards.length})</h3>
                <div className="flex flex-col gap-2 max-h-48 overflow-y-auto pr-2">
                  {hazards.map((h, i) => (
                    <div key={i} className="flex justify-between items-center bg-rose-50 border border-rose-100 p-3 rounded-lg">
                      <span className="font-bold text-rose-700 text-sm">{h.type}</span>
                      <span className="font-mono text-rose-500 text-xs bg-rose-100 px-2 py-1 rounded">{h.confidence}% Match</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Vision Viewer */}
          <div className="lg:col-span-2 flex flex-col justify-center items-center bg-slate-900 border border-slate-800 p-2 rounded-3xl shadow-2xl relative overflow-hidden min-h-[500px]">
            {!imagePreview ? (
              <div className="text-slate-600 font-mono text-sm tracking-widest uppercase animate-pulse">
                Awaiting Video Feed...
              </div>
            ) : (
              <div className="relative inline-block">
                {/* The uploaded image */}
                <img 
                  src={imagePreview} 
                  alt="Drone Feed" 
                  className="max-w-full max-h-[700px] rounded-2xl object-contain shadow-black/50 shadow-xl"
                  onLoad={onImageLoad}
                />
                
                {/* Overlaid Bounding Boxes */}
                {hazards.map((hazard, index) => {
                  const [x1, y1, x2, y2] = hazard.box;
                  // Calculate percentages based on the original image dimensions
                  const left = (x1 / imageSize.width) * 100;
                  const top = (y1 / imageSize.height) * 100;
                  const width = ((x2 - x1) / imageSize.width) * 100;
                  const height = ((y2 - y1) / imageSize.height) * 100;

                  return (
                    <div 
                      key={index}
                      className="absolute border-[3px] border-rose-500 bg-rose-500/20 shadow-[0_0_15px_rgba(225,29,72,0.8)]"
                      style={{
                        left: `${left}%`,
                        top: `${top}%`,
                        width: `${width}%`,
                        height: `${height}%`,
                      }}
                    >
                      {/* Box Label */}
                      <div className="absolute -top-6 left-[-3px] bg-rose-500 text-white text-[10px] font-bold font-mono px-2 py-1 uppercase whitespace-nowrap shadow-md">
                        {hazard.type} {hazard.confidence}%
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;