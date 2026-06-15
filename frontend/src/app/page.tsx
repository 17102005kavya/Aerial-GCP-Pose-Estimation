"use client";

import React, { useState, useEffect } from "react";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { UploadArea } from "@/components/UploadArea";
import { PredictionCard } from "@/components/PredictionCard";
import { ImageViewer } from "@/components/ImageViewer";
import { predictGcpImage, PredictResponse } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import confetti from "canvas-confetti";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Sparkles, AlertCircle, ArrowDown, HelpCircle, Layers, CheckCircle2, ChevronRight } from "lucide-react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Clear object URL to prevent memory leaks when file changes
  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  const handleFileSelected = (selectedFile: File) => {
    setFile(selectedFile);
    
    // Revoke previous URL if exists
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
    }
    
    setImageUrl(URL.createObjectURL(selectedFile));
    setPrediction(null);
  };

  const handleRunInference = async () => {
    if (!file) {
      toast.error("Please upload an image first.");
      return;
    }

    setIsLoading(true);
    const toastId = toast.loading("Uploading image and running PyTorch model cascade...");

    try {
      const response = await predictGcpImage(file);
      setPrediction(response);
      
      toast.dismiss(toastId);
      
      if (response.isSimulation) {
        toast.warning("Prediction loaded in offline simulation fallback mode.");
      } else {
        toast.success("Pose estimation completed successfully!");
      }

      // Celebratory confetti on success
      confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.8 },
        colors: ['#10b981', '#3b82f6', '#6366f1']
      });

    } catch (err: any) {
      console.error(err);
      toast.dismiss(toastId);
      toast.error(err.message || "Model prediction failed. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const scrollToWorkspace = () => {
    document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />

      <main className="flex-1 flex flex-col">
        {/* HERO SECTION */}
        <section className="relative overflow-hidden bg-background py-20 sm:py-24 border-b border-border/20">
          {/* Subtle grid background */}
          <div className="absolute inset-0 bg-[linear-gradient(to_right,#8080800b_1px,transparent_1px),linear-gradient(to_bottom,#8080800b_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />
          
          <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 relative text-center">
            <motion.div
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="flex justify-center mb-6"
            >
              <div className="inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-3.5 py-1 text-xs font-semibold text-primary backdrop-blur-sm">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Next-Gen Computer Vision Workspace</span>
              </div>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl md:text-6xl max-w-4xl mx-auto leading-[1.1]"
            >
              High-Precision GCP <span className="bg-gradient-to-r from-primary to-indigo-500 bg-clip-text text-transparent">Pose Estimation</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="mt-6 text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto leading-relaxed"
            >
              Automatically localize Ground Control Points in high-resolution aerial imagery and classify marker shapes with sub-pixel coordinate regression.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="mt-10 flex flex-wrap justify-center gap-4"
            >
              <Button onClick={scrollToWorkspace} size="lg" className="rounded-xl px-7 font-bold shadow-md shadow-primary/10 gap-2 hover:scale-[1.01] active:scale-[0.99] transition-transform">
                Start Detection
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button 
                variant="outline" 
                size="lg" 
                onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}
                className="rounded-xl px-7 font-semibold"
              >
                Learn Methodology
              </Button>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.7, duration: 1 }}
              className="mt-16 flex justify-center"
            >
              <button onClick={scrollToWorkspace} className="animate-bounce p-2 text-muted-foreground hover:text-foreground transition-colors" aria-label="Scroll down">
                <ArrowDown className="h-5 w-5" />
              </button>
            </motion.div>
          </div>
        </section>

        {/* WORKSPACE SECTION */}
        <section id="workspace" className="w-full py-16 sm:py-20 scroll-mt-16">
          <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col gap-3 mb-10 text-center md:text-left">
              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                Detection Workspace
              </h2>
              <p className="text-sm text-muted-foreground max-w-lg">
                Upload an aerial image, run the multi-stage cascade predictor, and interactively zoom or pan to verify coordinates.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              {/* Control Panel: Left (4 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-6 w-full">
                <UploadArea onFileSelected={handleFileSelected} disabled={isLoading} />
                
                <AnimatePresence>
                  {file && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden flex flex-col gap-4"
                    >
                      {!prediction && (
                        <Button
                          onClick={handleRunInference}
                          disabled={isLoading}
                          size="lg"
                          className="w-full rounded-xl font-bold shadow-md shadow-primary/5 hover:shadow-primary/10 gap-2 relative bg-primary hover:bg-primary/95 text-primary-foreground group"
                        >
                          <Play className="h-4.5 w-4.5 text-primary-foreground fill-primary-foreground/20 group-hover:scale-105 transition-transform" />
                          <span>{isLoading ? "Running EfficientNet Cascade..." : "Run Pose Estimator"}</span>
                        </Button>
                      )}

                      {prediction && (
                        <PredictionCard prediction={prediction} />
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Interactive Viewer: Right (7 cols) */}
              <div className="lg:col-span-7 w-full">
                {imageUrl && prediction ? (
                  <ImageViewer imageUrl={imageUrl} prediction={prediction} />
                ) : (
                  <div className="relative aspect-[4/3] w-full overflow-hidden border border-border/60 rounded-2xl bg-card flex flex-col items-center justify-center text-center p-6 shadow-sm">
                    {/* Subtle placeholder grid lines */}
                    <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808005_1px,transparent_1px),linear-gradient(to_bottom,#80808005_1px,transparent_1px)] bg-[size:16px_16px]" />
                    
                    <div className="relative z-10 flex flex-col items-center gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-muted/80 text-muted-foreground border border-border/40">
                        <Layers className="h-5.5 w-5.5 text-muted-foreground/60" />
                      </div>
                      <div className="flex flex-col gap-1 max-w-[280px]">
                        <h3 className="font-semibold text-sm text-card-foreground">
                          Spatial Visualizer Offline
                        </h3>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                          {file 
                            ? "Click 'Run Pose Estimator' to analyze the image and generate prediction target markers."
                            : "Upload a high-resolution drone photo to initiate the zoom-and-pan coordinate workspace."}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* METHODOLOGY GUIDE SECTION */}
        <section id="how-it-works" className="w-full bg-muted/30 border-t border-border/30 py-16 sm:py-20">
          <div className="max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                How AeroPoint AI Works
              </h2>
              <p className="text-sm text-muted-foreground mt-3 leading-relaxed">
                Under the hood, AeroPoint AI runs a state-of-the-art dual-task Deep Learning model utilizing a coarse-to-fine cropping cascade to achieve sub-pixel keypoint accuracy.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Card 1 */}
              <div className="bg-card border border-border/40 p-6 rounded-2xl flex flex-col gap-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
                  <span className="font-bold text-sm">1</span>
                </div>
                <h3 className="font-bold text-base text-card-foreground">
                  Dual-Task EfficientNet-B3
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  The model extracts rich feature grids via an EfficientNet-B3 backbone, branching into two heads: a keypoint regression head using a Sigmoid layer for normalized coordinates, and a shape classification head outputting 3-class logits (Cross, L-Shaped, Square).
                </p>
              </div>

              {/* Card 2 */}
              <div className="bg-card border border-border/40 p-6 rounded-2xl flex flex-col gap-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
                  <span className="font-bold text-sm">2</span>
                </div>
                <h3 className="font-bold text-base text-card-foreground">
                  Multi-Scale Training
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Rather than training only on tight crops, the model is trained with mixed scale choices (from full images down to tight 384px crops). This teaches the neural network to identify GCP shapes and locations at both coarse overview and fine-grained zoom levels.
                </p>
              </div>

              {/* Card 3 */}
              <div className="bg-card border border-border/40 p-6 rounded-2xl flex flex-col gap-4 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary border border-primary/20">
                  <span className="font-bold text-sm">3</span>
                </div>
                <h3 className="font-bold text-base text-card-foreground">
                  Coarse-to-Fine Cascade
                </h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  At inference, the system runs a cascade: Stage 0 processes the full image, and subsequent stages crop tighter square windows (1536px, 768px, 384px) centered on the previous prediction, resizing back to 512px to continuously resolve features with increasing precision.
                </p>
              </div>
            </div>

            {/* Performance Spec box */}
            <div className="mt-12 bg-card/60 border border-border/60 rounded-2xl p-6 flex flex-col sm:flex-row items-center gap-6 max-w-4xl mx-auto shadow-sm backdrop-blur-md">
              <CheckCircle2 className="h-10 w-10 text-emerald-500 shrink-0" />
              <div className="flex flex-col gap-1 text-center sm:text-left">
                <h4 className="font-bold text-sm text-foreground">
                  Sub-Pixel Pixel Accuracy Target (PCK@0.25)
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  By refining predictions through smaller crops, the model overcomes severe scale differences. The final output is capable of predicting GCP coordinates on 4000x3000 drone imagery within a tolerance of under 2.5 pixels.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
