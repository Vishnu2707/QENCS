"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { motion, AnimatePresence } from "framer-motion"

interface FocusIndicatorProps {
    score: number // 0 to 1
    label?: string
    isCalibrating?: boolean
}

export function FocusIndicator({ score, label = "Current Focus", isCalibrating = false }: FocusIndicatorProps) {
    const percentage = Math.round(score * 100)

    let strokeColor = "#3b82f6"; // Blue (Stable)
    let glowColor = "bg-blue-400";
    let textColor = "text-blue-600";

    if (score > 0.7) {
        strokeColor = "#10b981"; // Emerald
        glowColor = "bg-emerald-400";
        textColor = "text-emerald-600";
    } else if (score < 0.4) {
        strokeColor = "#f43f5e"; // Rose
        glowColor = "bg-rose-400";
        textColor = "text-rose-600";
    }

    return (
        <Card className="w-full relative overflow-hidden flex flex-col items-center bg-white/70 border-blue-100 shadow-xl min-h-[250px]">
            {/* Calibration Overlay */}
            <AnimatePresence>
                {isCalibrating && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 z-50 bg-white/40 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center"
                    >
                        <motion.div
                            animate={{ scale: [1, 1.05, 1] }}
                            transition={{ duration: 2, repeat: Infinity }}
                            className="w-16 h-16 rounded-full border-4 border-blue-500 border-t-transparent animate-spin mb-4"
                        />
                        <h4 className="text-blue-700 font-bold mb-2">Analyzing your neural baseline...</h4>
                        <p className="text-slate-600 text-xs">Stay still for 30s for personalization.</p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Glow effect background */}
            <div className={`absolute top-full left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-24 blur-[80px] opacity-20 rounded-t-full ${glowColor}`} />

            <CardHeader className="pb-0 text-center z-10 pt-6">
                <CardTitle className="text-xs font-bold uppercase tracking-widest text-slate-400">{label}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center justify-end h-[180px] pb-6 z-10 relative">
                <div className="relative w-[200px] h-[100px] overflow-hidden">
                    <svg className="w-full h-[200px]" viewBox="0 0 200 200">
                        {/* Background Arc */}
                        <path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke="#e2e8f0" // Slate-200
                            strokeWidth="15"
                            strokeLinecap="round"
                        />
                        {/* Progress Arc */}
                        <motion.path
                            d="M 20 100 A 80 80 0 0 1 180 100"
                            fill="none"
                            stroke={strokeColor}
                            strokeWidth="15"
                            strokeLinecap="round"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: score }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className="drop-shadow-md"
                        />
                    </svg>

                    {/* Value Text */}
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center transform translate-y-1">
                        <span className={`text-5xl font-bold ${textColor}`}>
                            {percentage}%
                        </span>
                    </div>
                </div>
                <span className="text-xs text-slate-500 font-semibold mt-4 uppercase tracking-wider">
                    {score > 0.7 ? "Flow State" : "Stable"}
                </span>
            </CardContent>
        </Card>
    )
}
