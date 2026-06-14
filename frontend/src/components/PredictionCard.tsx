"use client";

import { motion, Variants } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Target, Hash, Zap, Landmark, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { PredictResponse } from "@/lib/api";

interface PredictionCardProps {
  prediction: PredictResponse;
}

export function PredictionCard({ prediction }: PredictionCardProps) {
  const { x, y, shape, confidence, inference_time_ms, isSimulation } = prediction;

  const shapeColors: Record<string, string> = {
    "Cross": "from-teal-500 to-emerald-500 text-emerald-500",
    "L-Shaped": "from-amber-500 to-orange-500 text-orange-500",
    "Square": "from-blue-500 to-indigo-500 text-indigo-500"
  };

  const currentShapeColor = shapeColors[shape] || "from-primary to-indigo-500 text-primary";

  const cardVariants: Variants = {
    hidden: { opacity: 0, scale: 0.98 },
    visible: {
      opacity: 1,
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 150,
        damping: 15
      }
    }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={cardVariants}
      className="w-full"
    >
      <Card className="overflow-hidden border border-border/60 bg-card/50 backdrop-blur-md shadow-lg shadow-black/5 rounded-2xl relative">
        {/* Glow accent */}
        <div className={`absolute top-0 left-0 w-full h-[3px] bg-gradient-to-r ${currentShapeColor.split(' ')[0]} ${currentShapeColor.split(' ')[1]}`} />
        
        <CardHeader className="flex flex-row items-center justify-between pb-3">
          <CardTitle className="text-base font-bold tracking-tight text-card-foreground flex items-center gap-2">
            <Target className="h-4 w-4 text-primary" />
            Detection Results
          </CardTitle>
          
          {isSimulation && (
            <Badge variant="outline" className="bg-amber-500/10 text-amber-500 hover:bg-amber-500/15 border-amber-500/20 text-[10px] font-semibold flex items-center gap-1.5 py-0.5 px-2 rounded-full">
              <Info className="h-3 w-3" />
              <span>Simulation Fallback</span>
            </Badge>
          )}
        </CardHeader>
        
        <CardContent className="grid gap-5">
          {/* Main GCP Shape Visualizer */}
          <div className="flex items-center gap-4 bg-muted/30 dark:bg-muted/10 border border-border/40 rounded-xl p-4">
            <div className={`flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-tr ${currentShapeColor.split(' ')[0]} ${currentShapeColor.split(' ')[1]} text-white shadow-md`}>
              <span className="font-extrabold text-lg">
                {shape.charAt(0)}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider leading-none">
                Predicted GCP Shape
              </span>
              <span className="text-xl font-bold tracking-tight text-card-foreground">
                {shape}
              </span>
              <span className="text-xs text-muted-foreground mt-0.5">
                Confidence: <span className="font-semibold text-foreground">{(confidence * 100).toFixed(2)}%</span>
              </span>
            </div>
          </div>

          {/* Coordinates grid */}
          <div className="grid grid-cols-2 gap-4">
            <div className="border border-border/40 bg-background/40 rounded-xl p-3.5 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted text-muted-foreground border border-border/40">
                <Hash className="h-4.5 w-4.5" />
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider leading-none">
                  X Coordinate
                </span>
                <span className="text-lg font-bold tracking-tight text-foreground font-mono mt-0.5">
                  {x.toLocaleString()} <span className="text-xs text-muted-foreground font-sans font-medium">px</span>
                </span>
              </div>
            </div>

            <div className="border border-border/40 bg-background/40 rounded-xl p-3.5 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted text-muted-foreground border border-border/40">
                <Hash className="h-4.5 w-4.5" />
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider leading-none">
                  Y Coordinate
                </span>
                <span className="text-lg font-bold tracking-tight text-foreground font-mono mt-0.5">
                  {y.toLocaleString()} <span className="text-xs text-muted-foreground font-sans font-medium">px</span>
                </span>
              </div>
            </div>
          </div>

          {/* Inference Latency metadata */}
          <div className="flex items-center justify-between text-xs border-t border-border/30 pt-3">
            <span className="text-muted-foreground flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5 text-primary" />
              Inference Latency
            </span>
            <span className="font-semibold text-foreground font-mono">
              {inference_time_ms} ms
            </span>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
