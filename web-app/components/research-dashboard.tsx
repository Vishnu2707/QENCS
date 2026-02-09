
"use client"

import React, { useMemo, useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Float, ContactShadows, Environment } from '@react-three/drei'
import * as THREE from 'three'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Download, FlaskConical, LineChart as ChartIcon, Sigma, Zap, Globe, Brain, ListChecks } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend, ReferenceArea, AreaChart, Area
} from 'recharts'
import { Button } from '@/components/ui/button'


// 3D Scene for Hilbert Space Visualization
function HilbertSpace({ coords }: { coords: { x: number, y: number, z: number } }) {
    return (
        <>
            <ambientLight intensity={0.7} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <Environment preset="city" />

            {/* Grid Floor */}
            <gridHelper args={[10, 20, 0x94a3b8, 0xe2e8f0]} position={[0, -2, 0]} />

            {/* Simulation of the Bloch Sphere / Hilbert boundary */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry args={[1.9, 2, 64]} />
                <meshBasicMaterial color="#cbd5e1" transparent opacity={0.5} side={THREE.DoubleSide} />
            </mesh>

            {/* Current Quantum State */}
            <Float speed={3} rotationIntensity={0.2} floatIntensity={0.5}>
                <mesh position={[coords.x * 2, coords.y * 2, coords.z * 2]}>
                    <sphereGeometry args={[0.18, 32, 32]} />
                    <meshStandardMaterial color="#3b82f6" roughness={0.1} metalness={0.8} emissive="#3b82f6" emissiveIntensity={0.5} />
                </mesh>
            </Float>

            {/* Axis Lines */}
            <line>
                <bufferGeometry attach="geometry" {...new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-2.5, 0, 0), new THREE.Vector3(2.5, 0, 0)])} />
                <lineBasicMaterial attach="material" color="#94a3b8" />
            </line>
            <line>
                <bufferGeometry attach="geometry" {...new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, -2.5, 0), new THREE.Vector3(0, 2.5, 0)])} />
                <lineBasicMaterial attach="material" color="#94a3b8" />
            </line>

            {/* Axis Labels */}
            <Text position={[2.7, 0, 0]} fontSize={0.2} color="#64748b">|Z1⟩</Text>
            <Text position={[0, 2.7, 0]} fontSize={0.2} color="#64748b">|Z2⟩</Text>
            <Text position={[0, 0, 2.7]} fontSize={0.2} color="#64748b">|Z3⟩</Text>

            <ContactShadows resolution={1024} scale={10} blur={2} opacity={0.25} far={10} color="#000000" position={[0, -2, 0]} />
            <OrbitControls enableZoom={false} />
        </>
    )
}

interface ResearchDashboardProps {
    hilbertCoords: { x: number, y: number, z: number }
    lossHistory: { vqc: number, svm: number }[]
}

export function ResearchDashboard({ hilbertCoords, lossHistory }: ResearchDashboardProps) {
    const [hasMounted, setHasMounted] = useState(false)
    const [isLogsOpen, setIsLogsOpen] = useState(false)
    const [logs, setLogs] = useState<any[]>([])
    const [isLoadingLogs, setIsLoadingLogs] = useState(false)

    useEffect(() => setHasMounted(true), [])

    // MOCK DATA for sophisticated charts
    const publicationData = useMemo(() => {
        const years = Array.from({ length: 31 }, (_, i) => 1994 + i);
        return years.map(year => ({
            year,
            Overall: Math.pow(year - 1994, 2) * 2 + Math.random() * 100,
            USA: Math.pow(year - 1994, 1.8) * 1.5 + Math.random() * 50,
            China: year > 2005 ? Math.pow(year - 2005, 2.2) * 1.2 + Math.random() * 80 : Math.random() * 10,
            Germany: (year - 1994) * 15 + Math.random() * 30
        }));
    }, []);

    const decodingData = useMemo(() => {
        const points = 100;
        return Array.from({ length: points }, (_, i) => {
            const t = (i / points) * 8;
            // Grounded in EEG_data.csv ranges (Delta ~300k, Alpha ~30k, Beta ~45k)
            return {
                time: t.toFixed(2),
                'Battlefield': Math.round(Math.sin(t * 1.8) * 5000 + 45000 + (t > 2.2 && t < 3.8 ? 15000 : 0) + Math.random() * 5000),
                'Python': Math.round(Math.cos(t * 1.2) * 4000 + 33000 + (t > 3.8 && t < 5.8 ? 25000 : 0) + Math.random() * 4000),
                'Internal': Math.round(Math.sin(t * 2.5) * 6000 + 22000 + (t > 4 ? 8000 : 0) + Math.random() * 6000)
            };
        });
    }, []);

    const strategyData = useMemo(() => {
        return [1, 2, 3, 4, 5, 6].map(sub => {
            const points = 50;
            const baseline = 0.65 + Math.random() * 0.1;
            return {
                id: sub,
                name: `Subject No.${sub}`,
                data: Array.from({ length: points }, (_, i) => ({
                    n: i * 10,
                    // Simulating learning curves and algorithm convergence
                    s_rs: baseline + Math.sin(i / 10) * 0.05 + (i / points) * 0.05,
                    s_h: baseline - 0.05 + Math.cos(i / 12) * 0.04 + (i / points) * 0.12,
                    s_hr: baseline + 0.05 + (i * i / (points * points)) * 0.1 + Math.random() * 0.02
                }))
            };
        });
    }, []);

    const fetchLogs = async () => {
        setIsLoadingLogs(true)
        setIsLogsOpen(true)
        try {
            const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/research/logs`)
            if (res.ok) {
                const data = await res.json()
                setLogs(data.logs || [])
            }
        } catch (error) {
            console.error("Failed to fetch CNS logs:", error)
        } finally {
            setIsLoadingLogs(false)
        }
    }

    const handleDownloadBundle = async () => {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/research/export`);
        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = "qencs_research_bundle.zip";
            document.body.appendChild(a);
            a.click();
            a.remove();
        }
    }

    const lossChartData = lossHistory.map((d, i) => ({
        epoch: i,
        'Quantum VQC': d.vqc,
        'Classical SVM': d.svm
    }));

    if (!hasMounted) return <div className="bg-slate-50 animate-pulse" />;

    return (
        <div className="space-y-8 bg-slate-50 p-6 md:p-10 rounded-3xl border border-slate-200 text-slate-900 font-sans">

            {/* Header Section */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 pb-8 border-b border-slate-200">
                <div>
                    <div className="flex items-center gap-3 mb-2">
                        <div className="bg-blue-600 p-2 rounded-xl">
                            <FlaskConical className="w-6 h-6 text-white" />
                        </div>
                        <h2 className="text-3xl font-black tracking-tight text-slate-900">
                            Research Analytics Dashboard
                        </h2>
                    </div>
                    <div className="flex items-center gap-4">
                        <span className="text-[10px] bg-blue-100 text-blue-700 px-3 py-1 rounded-full font-bold border border-blue-200 uppercase tracking-widest">
                            Lab Mode v2.5
                        </span>
                        <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">
                            Quantum-Neural Enhancement & Cryptographic Studies
                        </p>
                    </div>
                </div>
                <div className="flex gap-3">
                    <Button
                        onClick={fetchLogs}
                        variant="outline"
                        className="rounded-full border-slate-200 text-slate-600 hover:bg-white hover:text-blue-600 px-6 font-bold text-xs"
                    >
                        View Raw CNS Logs
                    </Button>
                    <Button
                        onClick={handleDownloadBundle}
                        className="bg-blue-600 hover:bg-blue-700 text-white rounded-full flex items-center gap-2 px-8 font-bold text-xs shadow-lg shadow-blue-200"
                    >
                        <Download className="w-4 h-4" />
                        Export Research Bundle (ZIP)
                    </Button>
                </div>
            </div>

            {/* Top Visualization Bar */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

                {/* 3D State Projection */}
                <Card className="lg:col-span-7 bg-white border-slate-200 shadow-sm overflow-hidden h-[500px] flex flex-col group">
                    <CardHeader className="py-4 border-b border-slate-50 bg-slate-50/30">
                        <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                            <Zap className="w-3 h-3 text-blue-500" /> Quantum State Hilbert Projection
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 flex-1 relative">
                        <div className="absolute top-6 left-6 z-10 space-y-3 pointer-events-none">
                            <div className="bg-white/80 backdrop-blur-md p-3 rounded-xl border border-slate-100 shadow-sm">
                                <p className="text-[10px] font-black text-blue-600 uppercase mb-1">Observation Vector</p>
                                <div className="space-y-1">
                                    <div className="flex justify-between gap-8">
                                        <span className="text-[10px] font-mono text-slate-400">Position X</span>
                                        <span className="text-[10px] font-mono font-bold text-slate-700">{hilbertCoords.x.toFixed(6)}</span>
                                    </div>
                                    <div className="flex justify-between gap-8">
                                        <span className="text-[10px] font-mono text-slate-400">Position Y</span>
                                        <span className="text-[10px] font-mono font-bold text-slate-700">{hilbertCoords.y.toFixed(6)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <Canvas camera={{ position: [5, 5, 5], fov: 40 }}>
                            <HilbertSpace coords={hilbertCoords} />
                        </Canvas>
                    </CardContent>
                </Card>

                {/* Right Column: Convergence & Math */}
                <div className="lg:col-span-5 flex flex-col gap-6">
                    <Card className="flex-1 bg-white border-slate-200 shadow-sm flex flex-col overflow-hidden">
                        <CardHeader className="py-4 border-b border-slate-50 bg-slate-50/30">
                            <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                                <ChartIcon className="w-3 h-3 text-emerald-500" /> Convergence Benchmark
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-6 flex-1">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={lossChartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                    <XAxis dataKey="epoch" fontSize={10} tick={{ fill: '#94a3b8' }} hide />
                                    <YAxis fontSize={10} tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '12px', fontSize: '10px', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                    />
                                    <Legend verticalAlign="top" align="right" iconType="circle" wrapperStyle={{ fontSize: '10px', fontWeight: 'bold', paddingBottom: '20px' }} />
                                    <Line type="monotone" dataKey="Quantum VQC" stroke="#2563eb" strokeWidth={3} dot={false} animationDuration={1000} />
                                    <Line type="monotone" dataKey="Classical SVM" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>

                    <Card className="bg-white border-slate-200 shadow-sm overflow-hidden">
                        <CardHeader className="py-4 border-b border-slate-50 bg-slate-50/30">
                            <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                                <Sigma className="w-3 h-3 text-purple-500" /> Formal Model Architecture
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-6 space-y-4">
                            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 flex justify-center hover:bg-slate-100/50 transition-colors">
                                <BlockMath math="\psi(\theta) = \prod_{i=1}^{L} U_{ent}(\theta_i) R_x(\alpha_i) |0\rangle^{\otimes n}" />
                            </div>
                            <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 flex justify-center hover:bg-slate-100/50 transition-colors">
                                <InlineMath math="C(\theta) = \sum_{j} |y_j - \text{Tr}(\rho(\theta) O_j)| + \lambda \|\theta\|_2" />
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>

            {/* Research Metrics Bar */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                {[
                    { label: 'F1 Correlation', value: '0.892', change: '+4.2%', color: 'blue' },
                    { label: 'p-Value', value: '0.0034', change: 'σ=2.9', color: 'emerald' },
                    { label: 'Entanglement', value: '1.42 bit', change: 'MAX', color: 'purple' },
                    { label: 'Circuit Fidelity', value: '96.8%', change: '99.1% BC', color: 'rose' }
                ].map(stat => (
                    <div key={stat.label} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow group">
                        <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1 group-hover:text-blue-500 transition-colors">{stat.label}</p>
                        <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-black text-slate-800 tracking-tight">{stat.value}</span>
                            <span className={`text-[10px] font-bold text-${stat.color}-500`}>{stat.change}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Middle Section: Longitudinal Analysis (Figure 1 Replacement) */}
            <Card className="bg-white border-slate-200 shadow-sm overflow-hidden">
                <CardHeader className="py-4 border-b border-slate-200 bg-white">
                    <div className="flex justify-between items-center">
                        <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest">
                            <Globe className="w-4 h-4 text-blue-500" /> Longitudinal Publication Trend (1994-2024)
                        </CardTitle>
                        <div className="flex gap-4">
                            {['Overall', 'USA', 'China', 'Germany'].map((label, i) => (
                                <div key={label} className="flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ['#2563eb', '#10b981', '#f59e0b', '#ef4444'][i] }} />
                                    <span className="text-[10px] font-bold text-slate-500 uppercase">{label}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </CardHeader>
                <CardContent className="h-[400px] p-8">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={publicationData}>
                            <CartesianGrid strokeDasharray="1 1" stroke="#f1f5f9" />
                            <XAxis dataKey="year" fontSize={10} tick={{ fill: '#64748b' }} axisLine={{ stroke: '#e2e8f0' }} />
                            <YAxis fontSize={10} tick={{ fill: '#64748b' }} axisLine={{ stroke: '#e2e8f0' }} label={{ value: 'NUMBER OF PUBLICATIONS', angle: -90, position: 'insideLeft', style: { fontSize: '10px', fontWeight: 'bold', fill: '#94a3b8' } }} />
                            <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }} />
                            <Line type="monotone" dataKey="Overall" stroke="#2563eb" strokeWidth={3} dot={{ r: 2 }} />
                            <Line type="monotone" dataKey="USA" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} strokeDasharray="3 3" />
                            <Line type="monotone" dataKey="China" stroke="#f59e0b" strokeWidth={2} dot={{ r: 2 }} />
                            <Line type="monotone" dataKey="Germany" stroke="#ef4444" strokeWidth={2} dot={{ r: 2 }} strokeDasharray="5 5" />
                        </LineChart>
                    </ResponsiveContainer>
                    <div className="flex justify-around mt-4 text-[9px] font-black text-slate-400 uppercase tracking-widest">
                        <span>(A) Global Growth Index</span>
                        <span>(B) Territorial Contribution</span>
                        <span>(C) Neural-Quantum Correlation</span>
                    </div>
                </CardContent>
            </Card>

            {/* Bottom Row: Word Decoding & Strategy Comparison (Figure 3 & 4) */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">

                {/* Neural Word Decoding Analysis */}
                <Card className="bg-white border-blue-100 shadow-sm overflow-hidden flex flex-col">
                    <CardHeader className="border-b border-blue-50 bg-blue-50/20">
                        <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest text-center justify-center">
                            <Brain className="w-4 h-4 text-purple-500" /> Neural Word Decoding (Exp. Variance 12.4%)
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-6 h-[450px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={decodingData}>
                                <defs>
                                    <linearGradient id="colorWave" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.1} />
                                        <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                                <XAxis dataKey="time" fontSize={10} label={{ value: 'TIME (S)', position: 'insideBottom', offset: -5, style: { fontSize: '10px', fontWeight: 'bold', fill: '#94a3b8' } }} />
                                <YAxis fontSize={10} label={{ value: 'NORMALIZED FIRING RATE (HZ)', angle: -90, position: 'insideLeft', style: { fontSize: '10px', fontWeight: 'bold', fill: '#94a3b8' } }} />
                                <Tooltip />

                                {/* Experimental Phases */}
                                <ReferenceArea x1="0.5" x2="2.2" fill="#f8fafc" label={{ value: 'ITI', position: 'top', fontSize: 10, fill: '#94a3b8', fontWeight: 'bold' }} stroke="#cbd5e1" strokeDasharray="3 3" />
                                <ReferenceArea x1="2.2" x2="3.8" fill="#eff6ff" label={{ value: 'CUE', position: 'top', fontSize: 10, fill: '#3b82f6', fontWeight: 'bold' }} stroke="#3b82f6" strokeDasharray="3 3" />
                                <ReferenceArea x1="3.8" x2="5.8" fill="#f5f3ff" label={{ value: 'INTERNAL SPEECH', position: 'top', fontSize: 10, fill: '#8b5cf6', fontWeight: 'bold' }} stroke="#8b5cf6" strokeDasharray="3 3" />
                                <ReferenceArea x1="5.8" x2="7.8" fill="#ecfdf5" label={{ value: 'VOCALIZED', position: 'top', fontSize: 10, fill: '#10b981', fontWeight: 'bold' }} stroke="#10b981" strokeDasharray="3 3" />

                                <Area type="monotone" dataKey="Battlefield" stroke="#3b82f6" fillOpacity={0} strokeWidth={2} strokeDasharray="3 3" />
                                <Area type="monotone" dataKey="Python" stroke="#8b5cf6" fill="url(#colorWave)" strokeWidth={3} />
                                <Area type="monotone" dataKey="Internal" stroke="#f59e0b" fillOpacity={0} strokeWidth={2} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>

                {/* Strategy Performance Matrix */}
                <Card className="bg-white border-slate-200 shadow-sm overflow-hidden flex flex-col h-[450px]">
                    <CardHeader className="border-b border-slate-50 bg-slate-50/30">
                        <CardTitle className="text-xs font-black uppercase text-slate-400 flex items-center gap-2 tracking-widest text-center justify-center">
                            <ListChecks className="w-4 h-4 text-emerald-500" /> Performance of Different Strategies (n=6)
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4 flex-1">
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2 h-full">
                            {strategyData.map(subject => (
                                <div key={subject.id} className="border border-slate-100 rounded-lg p-2 flex flex-col bg-slate-50/30">
                                    <p className="text-[9px] font-black text-slate-500 mb-2 uppercase">{subject.name}</p>
                                    <div className="h-[120px] w-full">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={subject.data}>
                                                <XAxis dataKey="n" hide />
                                                <YAxis domain={['auto', 'auto']} hide />
                                                <Line type="step" dataKey="s_rs" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                                                <Line type="monotone" dataKey="s_h" stroke="#ef4444" strokeWidth={1.5} dot={false} />
                                                <Line type="monotone" dataKey="s_hr" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="flex justify-center gap-6 mt-4 pb-2">
                            {['s_rs', 's_h', 's_hr'].map((label, i) => (
                                <div key={label} className="flex items-center gap-2">
                                    <div className="w-3 h-0.5" style={{ backgroundColor: ['#3b82f6', '#ef4444', '#f59e0b'][i] }} />
                                    <span className="text-[9px] font-black italic text-slate-600">S_{label.split('_')[1]}</span>
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>

            </div>


            {/* CNS Logs Modal */}
            {isLogsOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/40 backdrop-blur-sm animate-in fade-in duration-200">
                    <div className="bg-white rounded-3xl border border-slate-200 shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col overflow-hidden">
                        <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                            <div>
                                <h3 className="text-lg font-black text-slate-900 flex items-center gap-2">
                                    <FlaskConical className="w-5 h-5 text-blue-600" /> Raw CNS Observation Logs
                                </h3>
                                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Real-time Hybrid Quantum-Neural Telemetry</p>
                            </div>
                            <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setIsLogsOpen(false)}
                                className="rounded-full h-8 w-8 p-0"
                            >
                                ✕
                            </Button>
                        </div>
                        <div className="flex-1 overflow-auto p-6 bg-slate-950 font-mono text-[11px]">
                            {isLoadingLogs ? (
                                <div className="text-blue-400 animate-pulse">Fetching latest quantum state vectors...</div>
                            ) : logs.length > 0 ? (
                                <div className="space-y-4">
                                    {logs.map((log, i) => (
                                        <div key={i} className="pb-4 border-b border-white/5 last:border-0">
                                            <div className="flex gap-4 mb-2">
                                                <span className="text-emerald-500">[{new Date().toLocaleTimeString()}]</span>
                                                <span className="text-blue-400">#FRAME_{i}</span>
                                            </div>
                                            <pre className="text-slate-300 leading-relaxed">
                                                {JSON.stringify(log, null, 2)}
                                            </pre>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-slate-500">No telemetry data captured in current session buffer.</div>
                            )}
                        </div>
                        <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex justify-end gap-3">
                            <Button
                                variant="outline"
                                onClick={() => setIsLogsOpen(false)}
                                className="rounded-full text-xs font-bold px-6 border-slate-200"
                            >
                                Close Window
                            </Button>
                            <Button
                                onClick={handleDownloadBundle}
                                className="bg-blue-600 hover:bg-blue-700 text-white rounded-full text-xs font-bold px-6"
                            >
                                <Download className="w-4 h-4 mr-2" /> Export Full Dataset
                            </Button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
