"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useState, useEffect } from "react";

const generateData = (pointCount: number) => {
    const data = [];
    for (let i = 0; i < pointCount; i++) {
        data.push({
            time: i,
            Theta: Math.random() * 10 + 5,
            Alpha: Math.random() * 8 + 8,
            Beta: Math.random() * 15 + 10,
        });
    }
    return data;
};

interface EEGChartProps {
    baselineValue?: number | null;
}

interface DataPoint {
    time: number;
    Theta: number;
    Alpha: number;
    Beta: number;
}

export function EEGChart({ baselineValue }: EEGChartProps) {
    const [data, setData] = useState<DataPoint[]>([]);
    const [mounted, setMounted] = useState(false);

    useEffect(() => {
        setMounted(true);
        setData(generateData(30));
        const interval = setInterval(() => {
            setData(prev => {
                if (prev.length === 0) return prev;
                const newData = [...prev.slice(1)];
                const lastTime = prev[prev.length - 1].time;
                newData.push({
                    time: lastTime + 1,
                    Theta: 10 + Math.sin(lastTime * 0.2) * 5 + Math.random() * 2,
                    Alpha: 15 + Math.cos(lastTime * 0.3) * 3 + Math.random() * 2,
                    Beta: 20 + Math.sin(lastTime * 0.5) * 8 + Math.random() * 3,
                });
                return newData;
            });
        }, 100);
        return () => clearInterval(interval);
    }, []);

    if (!mounted) return <div className="h-[350px] w-full bg-slate-100 rounded-xl animate-pulse" />;

    return (
        <Card className="col-span-2 border-blue-100 bg-white/50 backdrop-blur-md shadow-xl text-slate-900">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-xl font-bold text-slate-800">
                    Real-time EEG Monitor
                </CardTitle>
                <div className="flex space-x-2 text-xs font-medium">
                    <span className="flex items-center gap-1 text-purple-600"><div className="w-2 h-2 bg-purple-600 rounded-full" /> Theta</span>
                    <span className="flex items-center gap-1 text-blue-600"><div className="w-2 h-2 bg-blue-600 rounded-full" /> Alpha</span>
                    <span className="flex items-center gap-1 text-emerald-600"><div className="w-2 h-2 bg-emerald-600 rounded-full" /> Beta</span>
                </div>
            </CardHeader>
            <CardContent className="h-[320px] w-full pt-4 min-w-0">
                <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%" minHeight={300}>
                        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#cbd5e1" opacity={0.5} vertical={false} />
                            <XAxis
                                dataKey="time"
                                stroke="#64748b"
                                tick={false}
                                label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5, fill: '#64748b', fontSize: 12 }}
                            />
                            <YAxis
                                stroke="#64748b"
                                domain={[0, 40]}
                                tick={{ fontSize: 12 }}
                                tickLine={false}
                                axisLine={false}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)', color: '#0f172a' }}
                                itemStyle={{ fontSize: '12px', fontWeight: 'bold' }}
                                labelStyle={{ display: 'none' }}
                            />

                            {/* Baseline Marker (Phase 2) */}
                            {baselineValue && (
                                <ReferenceLine
                                    y={baselineValue * 30} // Mapping prob/ratio to chart Y-scale roughly for visual
                                    label={{ position: 'right', value: 'Baseline', fill: '#94a3b8', fontSize: 10 }}
                                    stroke="#94a3b8"
                                    strokeDasharray="5 5"
                                />
                            )}

                            <Line type="monotone" dataKey="Theta" stroke="#9333ea" strokeWidth={2} dot={false} isAnimationActive={false} />
                            <Line type="monotone" dataKey="Alpha" stroke="#2563eb" strokeWidth={2} dot={false} isAnimationActive={false} />
                            <Line type="monotone" dataKey="Beta" stroke="#059669" strokeWidth={2} dot={false} isAnimationActive={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
