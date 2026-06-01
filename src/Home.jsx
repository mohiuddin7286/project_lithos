import React from 'react';

function Home({ onLaunchDashboard }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100 animate-gradient-x text-slate-800 font-sans relative overflow-hidden selection:bg-blue-200">
      
      {/* Soft Background Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-blue-300/30 rounded-full blur-[120px] pointer-events-none"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[600px] h-[600px] bg-indigo-300/30 rounded-full blur-[120px] pointer-events-none"></div>

      <div className="max-w-7xl mx-auto px-6 py-12 relative z-10">
        
        {/* Navigation / Header */}
        <nav className="flex justify-between items-center mb-16">
          <div className="text-2xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-700 to-indigo-700">
            ⛰️ Project Lithos
          </div>
          <button 
            onClick={onLaunchDashboard}
            className="text-sm font-bold text-blue-600 hover:text-blue-800 transition-colors border-b-2 border-transparent hover:border-blue-600 pb-1"
          >
            Open Telemetry &rarr;
          </button>
        </nav>

        {/* Hero Section */}
        <div className="text-center max-w-4xl mx-auto mb-20 mt-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-100/50 border border-blue-200 text-blue-700 text-sm font-semibold mb-6 shadow-sm">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-blue-500"></span>
            </span>
            System Online v1.0
          </div>
          <h1 className="text-6xl md:text-7xl font-extrabold tracking-tight text-slate-800 mb-8 leading-tight">
            Predictive AI for <br/>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
              Slope Stability
            </span>
          </h1>
          <p className="text-xl text-slate-500 font-medium mb-10 leading-relaxed max-w-2xl mx-auto">
            Project Lithos is an enterprise-grade geotechnical monitoring engine. It fuses live sensor telemetry with LightGBM machine learning to predict rockfalls and slope failures before they happen.
          </p>
          
          <div className="flex justify-center gap-4">
            <button 
              onClick={onLaunchDashboard}
              className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-4 px-8 rounded-2xl transition-all duration-300 shadow-[0_10px_20px_rgba(79,70,229,0.3)] hover:shadow-[0_15px_25px_rgba(79,70,229,0.4)] hover:-translate-y-1 flex items-center gap-2"
            >
              Launch Dashboard
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
            <a href="#guide" className="bg-white/50 backdrop-blur-md border border-slate-200 text-slate-700 hover:bg-white/80 font-bold py-4 px-8 rounded-2xl transition-all duration-300 shadow-sm hover:shadow-md">
              How it works
            </a>
          </div>
        </div>

        {/* What it is used for (Features) */}
        <div className="grid md:grid-cols-3 gap-8 mb-24">
          <FeatureCard 
            icon="📡"
            title="Live Telemetry Fusion"
            desc="Aggregates data from multi-point field sensors, including displacement radars, piezometers, and rain gauges in real-time."
          />
          <FeatureCard 
            icon="🧠"
            title="LightGBM Inference"
            desc="Translates highly heterogeneous, multi-source data streams into a single, high-fidelity risk probability score using ensemble learning."
          />
          <FeatureCard 
            icon="🚨"
            title="Automated Alerting"
            desc="Triggers localized evacuation protocols and dispatches automated drone inspections when structural failure probabilities exceed 75%."
          />
        </div>

        {/* How to use it (Guide) */}
        <div id="guide" className="bg-white/70 backdrop-blur-xl border border-white p-10 rounded-3xl shadow-xl">
          <h2 className="text-3xl font-extrabold text-slate-800 mb-8 text-center">Operational Guide</h2>
          <div className="grid md:grid-cols-3 gap-12">
            <Step 
              num="01" 
              title="Calibrate Sensors" 
              desc="Use the left panel of the dashboard to sync the UI with your current physical sensor readings (Displacement, Pore Pressure, Rainfall)." 
            />
            <Step 
              num="02" 
              title="Run Diagnostics" 
              desc="Click the primary CTA button. The React frontend will package the JSON payload and POST it to the Python AI Core." 
            />
            <Step 
              num="03" 
              title="Monitor Risk" 
              desc="The AI will instantly return a Factor of Safety (FS) prediction. Watch the UI glow Green (Safe), Yellow (Warning), or Red (Critical)." 
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-20 text-slate-400 text-sm font-medium">
          &copy; 2026 Project Lithos. Engineered for Open-Pit Mine Safety.
        </footer>
      </div>
    </div>
  );
}

// Reusable UI Components for the Home Page
function FeatureCard({ icon, title, desc }) {
  return (
    <div className="bg-white/80 backdrop-blur-xl border border-white p-8 rounded-3xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
      <div className="text-4xl mb-4 bg-blue-50 w-16 h-16 rounded-2xl flex items-center justify-center shadow-inner border border-blue-100">{icon}</div>
      <h3 className="text-xl font-bold text-slate-800 mb-3">{title}</h3>
      <p className="text-slate-500 font-medium leading-relaxed">{desc}</p>
    </div>
  );
}

function Step({ num, title, desc }) {
  return (
    <div className="relative">
      <div className="text-6xl font-black text-slate-100 absolute -top-8 -left-4 z-0 drop-shadow-sm">{num}</div>
      <div className="relative z-10">
        <h4 className="text-xl font-bold text-slate-800 mb-2">{title}</h4>
        <p className="text-slate-500 font-medium">{desc}</p>
      </div>
    </div>
  );
}

export default Home;