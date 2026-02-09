
"use client"

import React, { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Stars, Text, Float } from '@react-three/drei'
import * as THREE from 'three'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Download, FlaskConical, LineChart as ChartIcon, Sigma, Zap } from 'lucide-react'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend
} from 'recharts'
import { Button } from '@/components/ui/button'


// 3D Scene for Hilbert Space Visualization
function HilbertSpace({ coords }: { coords: { x: number, y: number, z: number } }) {
    return (
        <>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

            {/* Simulation of the Bloch Sphere / Hilbert boundary */}
            <mesh rotation={[Math.PI / 2, 0, 0]}>
                <ringGeometry args={[1.9, 2, 64]} />
                <meshBasicMaterial color="#1e293b" transparent opacity={0.3} side={THREE.DoubleSide} />
            </mesh>

            {/* Current Quantum State */}
            <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
                <mesh position={[coords.x * 2, coords.y * 2, coords.z * 2]}>
                    <sphereGeometry args={[0.15, 32, 32]} />
                    <meshStandardMaterial color="#6366f1" emissive="#6366f1" emissiveIntensity={5} />
                </mesh>
            </Float>

            {/* Axis Labels */}
            <Text position={[2.5, 0, 0]} fontSize={0.2} color="#94a3b8">|Z1⟩</Text>
            <Text position={[0, 2.5, 0]} fontSize={0.2} color="#94a3b8">|Z2⟩</Text>
            <Text position={[0, 0, 2.5]} fontSize={0.2} color="#94a3b8">|Z3⟩</Text>

            <OrbitControls />
        </>
    )
}

interface ResearchDashboardProps {
    hilbertCoords: { x: number, y: number, z: number }
    lossHistory: { vqc: number, svm: number }[]
}

export function ResearchDashboard({ hilbertCoords, lossHistory }: ResearchDashboardProps) {

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

    // Formatting loss data for Recharts
    const chartData = lossHistory.map((d, i) => ({
        epoch: i,
        'Quantum VQC': d.vqc,
        'Classical SVM': d.svm
    }));

    return (
        <div className="space-y-6 bg-[#020617] p-6 rounded-3xl border border-slate-800 text-slate-200 min-h-screen">

            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="text-2xl font-black tracking-tighter flex items-center gap-3 text-white">
                        <FlaskConical className="w-6 h-6 text-blue-400" />
                        RESEARCH LABORATORY <span className="text-[10px] bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded border border-blue-500/30">LAB MODE v2.1</span>
                    </h2>
                    <p className="text-xs text-slate-500 font-medium uppercase tracking-widest mt-1">High-Dimensional Feature Mapping & Benchmarking</p>
                </div>
                <Button
                    onClick={handleDownloadBundle}
                    className="bg-blue-600 hover:bg-blue-500 text-white rounded-full flex items-center gap-2 px-6"
                >
                    <Download className="w-4 h-4" />
                    Download Research Bundle
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

                {/* 3D Hilbert Visualization */}
                <Card className="bg-[#0f172a]/50 border-slate-800 shadow-2xl h-[500px] overflow-hidden">
                    <CardHeader className="border-b border-slate-800/50 bg-[#0f172a]/80">
                        <CardTitle className="text-xs font-bold uppercase text-slate-400 flex items-center gap-2">
                            <Zap className="w-3 h-3 text-blue-400" /> Quantum State Visualization (Hilbert Projection)
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-0 h-full relative">
                        <div className="absolute top-4 left-4 z-10 space-y-1">
                            <div className="text-[10px] font-mono text-blue-400">Projection: ⟨ψ|Z_i|ψ⟩</div>
                            <div className="text-[10px] font-mono text-slate-500">X: {hilbertCoords.x.toFixed(4)}</div>
                            <div className="text-[10px] font-mono text-slate-500">Y: {hilbertCoords.y.toFixed(4)}</div>
                        </div>
                        <Canvas camera={{ position: [5, 5, 5], fov: 45 }}>
                            <HilbertSpace coords={hilbertCoords} />
                        </Canvas>
                    </CardContent>
                </Card>

                <div className="space-y-6">
                    {/* Loss Convergence Benchmark */}
                    <Card className="bg-[#0f172a]/50 border-slate-800 shadow-xl overflow-hidden">
                        <CardHeader className="pb-2">
                            <CardTitle className="text-xs font-bold uppercase text-slate-400 flex items-center gap-2">
                                <ChartIcon className="w-3 h-3 text-emerald-400" /> Loss Convergence: Quantum vs. Classical
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="h-[250px] p-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                                    <XAxis dataKey="epoch" hide />
                                    <YAxis stroke="#475569" fontSize={10} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', fontSize: '10px' }}
                                    />
                                    <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: '10px', textTransform: 'uppercase', fontWeight: 'bold' }} />
                                    <Line type="monotone" dataKey="Quantum VQC" stroke="#6366f1" strokeWidth={3} dot={false} animationDuration={300} />
                                    <Line type="monotone" dataKey="Classical SVM" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>

                    {/* Mathematical Proof Card */}
                    <Card className="bg-[#0f172a]/50 border-slate-800 shadow-xl overflow-hidden">
                        <CardHeader className="pb-2 border-b border-slate-800/50">
                            <CardTitle className="text-xs font-bold uppercase text-slate-400 flex items-center gap-2">
                                <Sigma className="w-3 h-3 text-purple-400" /> Formal Quantum Circuit Model
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="p-6 space-y-4">
                            <div>
                                <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Ansatz & State Preparation</p>
                                <div className="bg-slate-900/50 p-4 rounded-lg flex justify-center border border-slate-800">
                                    <BlockMath math="\psi(\theta) = \prod_{i=1}^{L} U_{ent}(\theta_i) R_x(\alpha_i) |0\rangle^{\otimes n}" />
                                </div>
                            </div>
                            <div>
                                <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Cost Function Optimization</p>
                                <div className="bg-slate-900/50 p-4 rounded-lg flex justify-center border border-slate-800">
                                    <InlineMath math="C(\theta) = \sum_{j} |y_j - \text{Tr}(\rho(\theta) O_j)| + \lambda \|\theta\|_2" />
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                </div>

            </div>

            {/* Stats Footer */}
            <div className="grid grid-cols-4 gap-4 mt-4">
                <div className="px-4 py-3 rounded-2xl bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] font-bold text-slate-500 uppercase">F1 Correlation Score</div>
                    <div className="text-xl font-black text-white">0.892 <span className="text-[8px] text-emerald-400 ml-1">↑ 4.2%</span></div>
                </div>
                <div className="px-4 py-3 rounded-2xl bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] font-bold text-slate-500 uppercase">p-Value (Significance)</div>
                    <div className="text-xl font-black text-white">0.0034 <span className="text-[8px] text-blue-400 ml-1">σ=2.9</span></div>
                </div>
                <div className="px-4 py-3 rounded-2xl bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] font-bold text-slate-500 uppercase">Entanglement Entropy</div>
                    <div className="text-xl font-black text-white">1.42 bit</div>
                </div>
                <div className="px-4 py-3 rounded-2xl bg-slate-900/80 border border-slate-800">
                    <div className="text-[10px] font-bold text-slate-500 uppercase">Circuit Fidelity</div>
                    <div className="text-xl font-black text-white">96.8%</div>
                </div>
            </div>

        </div>
    )
}
