"use client"

import { EEGChart } from "@/components/eeg-chart";
import { FocusIndicator } from "@/components/focus-indicator";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Brain, Zap, Clock, BarChart3, Settings,
  CheckCircle, PieChart as PieChartIcon,
  Target, AlertTriangle, ShieldCheck, TrendingUp, FlaskConical
} from "lucide-react";
import { useState, useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import { cn } from "@/lib/utils";
import { useEEGData } from "@/hooks/use-eeg-data";
import { motion, AnimatePresence } from "framer-motion";

const ResearchDashboard = dynamic(
  () => import("@/components/research-dashboard").then((mod) => mod.ResearchDashboard),
  { ssr: false }
);
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip,
  ReferenceLine
} from 'recharts';

const subjects = [
  { id: "S001", name: "Subject Alpha", status: "Active" },
  { id: "S002", name: "Subject Beta", status: "Idle" },
  { id: "S003", name: "Subject Gamma", status: "Offline" },
];

export default function Home() {
  const [activeSubject, setActiveSubject] = useState("S001");
  const [activeTab, setActiveTab] = useState<"dashboard" | "settings" | "research">("dashboard");
  const {
    currentFocus, advice, lapseProb, status, isCalibrating, baselineValue,
    entropy, bandPower, confidence, interventions,
    hilbertCoords, lossHistory,
    sensitivity, setSensitivity
  } = useEEGData();

  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [wasCalibrating, setWasCalibrating] = useState(true);
  const [focusTime, setFocusTime] = useState(0);
  const [hasMounted, setHasMounted] = useState(false);

  useEffect(() => {
    setHasMounted(true);
  }, []);

  // Tracking focus time (mocked accumulation)
  useEffect(() => {
    if (status === 'active' && !isCalibrating && currentFocus > 0.7) {
      setFocusTime(prev => prev + 2);
    }
  }, [currentFocus, status, isCalibrating]);

  // Toast Logic
  useEffect(() => {
    if (wasCalibrating && !isCalibrating) {
      setToastMessage("Neural Profile Created. Personalized Coaching Active.");
      setShowToast(true);
      setTimeout(() => setShowToast(false), 5000);
    }
    setWasCalibrating(isCalibrating);
  }, [isCalibrating, wasCalibrating]);

  const handleSensitivityChange = (val: number) => {
    setSensitivity(val);
    setToastMessage(`Intervention Sensitivity set to ${Math.round(val * 100)}%`);
    setShowToast(true);
    setTimeout(() => setShowToast(false), 3000);
  };

  // Pie Chart Data
  const pieData = useMemo(() => [
    { name: 'Theta', value: bandPower.theta, color: '#9333ea' },
    { name: 'Alpha', value: bandPower.alpha, color: '#2563eb' },
    { name: 'Task Engagement', value: bandPower.beta, color: '#059669' },
  ], [bandPower]);

  // Bar Chart Data (Confidence)
  const confidenceData = useMemo(() => [
    { name: 'Decision Certainty', value: confidence * 100 }
  ], [confidence]);

  return (
    <div className="flex bg-slate-50 text-slate-900 font-sans relative">

      {/* Toast Notification */}
      <AnimatePresence>
        {showToast && (
          <motion.div
            initial={{ opacity: 0, y: 50, x: "-50%" }}
            animate={{ opacity: 1, y: 0, x: "-50%" }}
            exit={{ opacity: 0, y: 20, x: "-50%" }}
            className="fixed bottom-10 left-1/2 z-[100] bg-slate-900 text-white px-6 py-3 rounded-full shadow-2xl flex items-center space-x-3"
          >
            <CheckCircle className="w-5 h-5 text-emerald-400" />
            <span className="font-bold">{toastMessage}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <aside className="w-64 border-r border-blue-100 bg-white/70 backdrop-blur-xl flex flex-col hidden md:flex fixed h-full z-20 shadow-sm">
        <div className="p-6 border-b border-blue-50">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-md">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight text-slate-800">QENCS</span>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-4">
          <div className="px-4 mb-2 text-xs font-bold text-slate-400 uppercase tracking-wider">Navigation</div>
          <nav className="space-y-1 px-2 mb-6">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={cn(
                "w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all",
                activeTab === "dashboard" ? "bg-blue-50 text-blue-700 shadow-sm" : "text-slate-500 hover:bg-slate-50"
              )}
            >
              <BarChart3 className="w-4 h-4" />
              <span>Dashboard</span>
            </button>
            <button
              onClick={() => setActiveTab("settings")}
              className={cn(
                "w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all",
                activeTab === "settings" ? "bg-blue-50 text-blue-700 shadow-sm" : "text-slate-500 hover:bg-slate-50"
              )}
            >
              <Settings className="w-4 h-4" />
              <span>Settings</span>
            </button>
            <button
              onClick={() => setActiveTab("research")}
              className={cn(
                "w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-all text-left",
                activeTab === "research" ? "bg-slate-900 text-white shadow-md" : "text-slate-500 hover:bg-slate-50"
              )}
            >
              <FlaskConical className="w-4 h-4 text-blue-500" />
              <span>Research Analytics</span>
            </button>
          </nav>

          <div className="px-4 mb-2 text-xs font-bold text-slate-400 uppercase tracking-wider">Subject Profiles</div>
          <nav className="space-y-1 px-2">
            {subjects.map((sub) => (
              <button
                key={sub.id}
                onClick={() => setActiveSubject(sub.id)}
                className={cn(
                  "w-full flex items-center justify-between px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                  activeSubject === sub.id
                    ? "bg-slate-100 text-slate-900"
                    : "text-slate-500 hover:bg-slate-50 hover:text-slate-700"
                )}
              >
                <div className="flex items-center space-x-3">
                  <span className={cn("w-2 h-2 rounded-full", sub.status === 'Active' ? 'bg-emerald-500' : 'bg-slate-300')} />
                  <span>{sub.name}</span>
                </div>
              </button>
            ))}
          </nav>
        </div>

        <div className="p-4 border-t border-blue-50 bg-white/50">
          <div className="p-3 rounded-lg flex items-center space-x-3 border border-blue-100 bg-white shadow-sm">
            <div className={`w-2.5 h-2.5 rounded-full ${status === 'active' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
            <div className="flex flex-col">
              <span className={`text-xs font-bold ${status === 'active' ? 'text-emerald-700' : 'text-rose-700'}`}>
                {status === 'active' ? (isCalibrating ? 'Calibrating...' : 'Secure Stream') : 'System Offline'}
              </span>
              <span className="text-[10px] text-slate-400">Quantum Node: Active</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 md:ml-64 p-8 space-y-8 overflow-y-auto">

        {activeTab === "dashboard" ? (
          <>
            {/* Top Header Area */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <h1 className="text-3xl font-bold text-slate-900 tracking-tight">System Overview</h1>
                <p className="text-slate-500 font-medium">Session analysis for <span className="text-blue-600 font-semibold">{subjects.find(s => s.id === activeSubject)?.name}</span></p>
              </div>
              <div className="flex space-x-3">
                <Badge label={`Entropy: ${entropy.toFixed(2)}`} color="bg-orange-100 text-orange-700 border-orange-200" dotColor="bg-orange-500" />
                <Badge label={`Logic Sensitivity: ${Math.round(sensitivity * 100)}%`} color="bg-blue-100 text-blue-700 border-blue-200" dotColor="bg-blue-500" />
              </div>
            </div>

            {/* Top Data Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="flex flex-col gap-6 lg:col-span-1">
                <div className="flex-1">
                  <FocusIndicator score={currentFocus} isCalibrating={isCalibrating} />
                </div>
                <Card className="border-purple-200 bg-gradient-to-br from-purple-50 to-white shadow-lg relative overflow-hidden flex flex-col justify-center min-h-[160px]">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-purple-700 flex items-center gap-2 text-sm uppercase tracking-widest">
                      <Zap className="w-4 h-4" />
                      Quantum Logic Tip
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-lg font-bold text-slate-800 leading-tight">
                      &quot;{advice}&quot;
                    </p>
                  </CardContent>
                </Card>
              </div>

              <div className="lg:col-span-2 h-full">
                <EEGChart baselineValue={baselineValue} />
              </div>
            </div>

            {/* Neural Analytics Grid (Phase 3) */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 auto-rows-fr">
              {/* Card 1: Spectral Distribution */}
              <Card className="bg-white border-blue-50 shadow-sm overflow-hidden flex flex-col">
                <CardHeader className="pb-0">
                  <CardTitle className="text-xs font-bold uppercase text-slate-400 tracking-tighter flex items-center gap-2">
                    <PieChartIcon className="w-3 h-3" /> Spectral Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-center pt-0 min-h-[220px]">
                  {hasMounted ? (
                    <div className="h-[180px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={pieData}
                            innerRadius={50}
                            outerRadius={70}
                            paddingAngle={5}
                            dataKey="value"
                            animationDuration={500}
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <RechartsTooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[180px] w-full flex items-center justify-center bg-slate-50/50 animate-pulse rounded-lg" />
                  )}
                  <div className="flex justify-center gap-4 text-[10px] font-bold">
                    {pieData.map(item => (
                      <div key={item.name} className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                        <span className="text-slate-600">{item.name}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Card 2: Quantum Decision Space */}
              <Card className="bg-white border-blue-50 shadow-sm flex flex-col">
                <CardHeader>
                  <CardTitle className="text-xs font-bold uppercase text-slate-400 tracking-tighter flex items-center gap-2">
                    <Target className="w-3 h-3" /> Quantum Certainty
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-center pt-0 min-h-[220px]">
                  {hasMounted ? (
                    <div className="h-[140px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={confidenceData}>
                          <XAxis dataKey="name" hide />
                          <YAxis domain={[0, 100]} hide />
                          <Bar dataKey="value" fill="#6366f1" radius={[8, 8, 8, 8]} barSize={40} />
                          <ReferenceLine y={50} stroke="#e2e8f0" strokeDasharray="3 3" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="h-[140px] w-full flex items-center justify-center bg-slate-50/50 animate-pulse rounded-lg" />
                  )}
                  <div className="text-center mt-4">
                    <span className="text-2xl font-black text-slate-800">{Math.round(confidence * 100)}%</span>
                    <p className="text-[10px] text-slate-400 font-bold uppercase">Decision Score</p>
                  </div>
                </CardContent>
              </Card>

              {/* Card 3: Session Intelligence */}
              <Card className="bg-white border-blue-50 shadow-sm flex flex-col">
                <CardHeader>
                  <CardTitle className="text-xs font-bold uppercase text-slate-400 tracking-tighter">
                    Session Intelligence
                  </CardTitle>
                </CardHeader>
                <CardContent className="flex-1 flex flex-col justify-around min-h-[220px]">
                  <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <Clock className="w-4 h-4 text-blue-600" />
                      <span className="text-xs font-bold text-slate-600 uppercase">Focus Time</span>
                    </div>
                    <span className="text-sm font-bold text-blue-700">{Math.floor(focusTime / 60)}m {focusTime % 60}s</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <TrendingUp className="w-4 h-4 text-emerald-600" />
                      <span className="text-xs font-bold text-slate-600 uppercase">Avg Deviation</span>
                    </div>
                    <span className="text-sm font-bold text-emerald-700">{baselineValue ? Math.abs((lapseProb - baselineValue)).toFixed(2) : "0.00"}</span>
                  </div>
                  <div className="flex items-center justify-between p-3 bg-rose-50 rounded-lg">
                    <div className="flex items-center gap-3">
                      <AlertTriangle className="w-4 h-4 text-rose-600" />
                      <span className="text-xs font-bold text-slate-600 uppercase">Interventions</span>
                    </div>
                    <span className="text-sm font-bold text-rose-700">{interventions}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </>
        ) : activeTab === "research" ? (
          <ResearchDashboard hilbertCoords={hilbertCoords} lossHistory={lossHistory} />
        ) : (
          <div className="max-w-2xl mx-auto py-10">
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-3">
              <Settings className="w-6 h-6 text-blue-600" /> System Settings
            </h2>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Coach Intervention Sensitivity</CardTitle>
                <p className="text-sm text-slate-500">How quickly the Logic Agent should intervene when a lapse is detected.</p>
              </CardHeader>
              <CardContent className="space-y-6 pt-4">
                <div className="grid grid-cols-3 gap-4">
                  {[0.10, 0.15, 0.20].map((val) => (
                    <button
                      key={val}
                      onClick={() => handleSensitivityChange(val)}
                      className={cn(
                        "p-6 rounded-xl border-2 transition-all flex flex-col items-center gap-2",
                        sensitivity === val
                          ? "border-blue-600 bg-blue-50 shadow-md"
                          : "border-slate-100 hover:border-slate-200"
                      )}
                    >
                      <span className="font-black text-xl">{val * 100}%</span>
                      <span className="text-[10px] font-bold uppercase text-slate-400">
                        {val === 0.10 ? "High Frequency" : val === 0.15 ? "Standard" : "Low Impact"}
                      </span>
                    </button>
                  ))}
                </div>

                <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 flex items-start gap-4">
                  <ShieldCheck className="w-6 h-6 text-blue-500 flex-shrink-0" />
                  <div>
                    <p className="text-xs font-bold text-slate-800 uppercase">Personalization Guarantee</p>
                    <p className="text-xs text-slate-500 mt-1">Interventions are always processed relative to your neural baseline. Changing sensitivity shifts the peak-detection threshold.</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}

function Badge({ label, color, dotColor }: { label: string, color: string, dotColor: string }) {
  return (
    <div className={cn("flex items-center space-x-2 px-3 py-1.5 rounded-full border shadow-sm", color)}>
      <div className={cn("w-2 h-2 rounded-full", dotColor)} />
      <span className="text-[10px] font-black uppercase tracking-wider">{label}</span>
    </div>
  )
}
