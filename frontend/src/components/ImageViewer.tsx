"use client";

import React, { useState, useRef, useEffect } from "react";
import { ZoomIn, ZoomOut, Maximize2, MousePointer, Layers, Target, HelpCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { PredictResponse } from "@/lib/api";

interface ImageViewerProps {
  imageUrl: string;
  prediction: PredictResponse;
}

export function ImageViewer({ imageUrl, prediction }: ImageViewerProps) {
  const { x, y, shape, confidence, width: origW, height: origH, stages } = prediction;

  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [cursorCoords, setCursorCoords] = useState<{ x: number; y: number } | null>(null);
  const [showStages, setShowStages] = useState(false);
  const [imgLayout, setImgLayout] = useState({ width: 0, height: 0, left: 0, top: 0 });

  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // Normalize predicted target coordinates
  const xNorm = x / origW;
  const yNorm = y / origH;

  // Track layout of image on screen to place overlays perfectly
  const updateImageLayout = () => {
    if (imgRef.current) {
      setImgLayout({
        width: imgRef.current.clientWidth,
        height: imgRef.current.clientHeight,
        left: imgRef.current.offsetLeft,
        top: imgRef.current.offsetTop,
      });
    }
  };

  useEffect(() => {
    updateImageLayout();
    window.addEventListener("resize", updateImageLayout);
    return () => window.removeEventListener("resize", updateImageLayout);
  }, [imageUrl]);

  // Handle Zoom Wheel
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = 0.15;
    const direction = e.deltaY < 0 ? 1 : -1;
    const newScale = Math.max(1, Math.min(scale + direction * zoomFactor * scale, 15));
    
    setScale(newScale);
    if (newScale === 1) {
      setPosition({ x: 0, y: 0 });
    }
  };

  // Handle Mouse Drag for Pan
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    if (scale <= 1) return; // Only pan when zoomed in
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
    trackCursorPosition(e);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Track cursor location on original image resolution
  const trackCursorPosition = (e: React.MouseEvent) => {
    if (!imgRef.current) return;
    
    const rect = imgRef.current.getBoundingClientRect();
    
    // Position of cursor relative to image element (accounts for zoom scale)
    const clientXOnImg = e.clientX - rect.left;
    const clientYOnImg = e.clientY - rect.top;
    
    const u = clientXOnImg / rect.width;
    const v = clientYOnImg / rect.height;
    
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) {
      const origX = Math.round(u * origW);
      const origY = Math.round(v * origH);
      setCursorCoords({ x: origX, y: origY });
    } else {
      setCursorCoords(null);
    }
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setCursorCoords(null);
  };

  // Button Zoom Helpers
  const zoomIn = () => {
    setScale(prev => Math.min(prev + 1.5, 15));
  };

  const zoomOut = () => {
    setScale(prev => {
      const next = Math.max(prev - 1.5, 1);
      if (next === 1) setPosition({ x: 0, y: 0 });
      return next;
    });
  };

  const resetZoom = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  // Compute marker center
  const markerLeft = imgLayout.left + xNorm * imgLayout.width;
  const markerTop = imgLayout.top + yNorm * imgLayout.height;

  // Stage borders styles (colors corresponding to each stage cascade)
  const stageColors = [
    "border-cyan-500/80 bg-cyan-500/5",     // Stage 0 (Full)
    "border-purple-500/80 bg-purple-500/5", // Stage 1 (1536)
    "border-amber-500/80 bg-amber-500/5",   // Stage 2 (768)
    "border-rose-500/80 bg-rose-500/5",     // Stage 3 (384)
  ];

  return (
    <div className="flex flex-col gap-3 w-full">
      {/* Top toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowStages(!showStages)}
            className={`h-8 gap-1.5 rounded-lg text-xs font-semibold ${showStages ? 'bg-primary/10 border-primary text-primary hover:bg-primary/15' : ''}`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>{showStages ? "Hide Cascade Stages" : "Show Cascade Stages"}</span>
          </Button>
        </div>

        {/* Real-time coordinates */}
        <div className="h-8 flex items-center">
          {cursorCoords ? (
            <Badge variant="secondary" className="font-mono text-[10px] tracking-tight bg-muted/80 text-muted-foreground px-2.5 py-1 flex items-center gap-1 border border-border/40 rounded-lg">
              <MousePointer className="h-3 w-3 text-primary" />
              <span>X: {cursorCoords.x} px</span>
              <span className="text-muted-foreground/40">|</span>
              <span>Y: {cursorCoords.y} px</span>
            </Badge>
          ) : (
            <span className="text-[10px] text-muted-foreground font-medium flex items-center gap-1.5 px-2">
              <HelpCircle className="h-3.5 w-3.5" />
              Hover image to track coordinates
            </span>
          )}
        </div>
      </div>

      {/* Main interactive viewport */}
      <div 
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        className={`relative aspect-[4/3] w-full overflow-hidden border border-border/60 rounded-2xl bg-slate-950/60 shadow-inner flex items-center justify-center select-none ${
          scale > 1 ? (isDragging ? "cursor-grabbing" : "cursor-grab") : "cursor-default"
        }`}
      >
        {/* Transforming viewer box */}
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: "center center",
            transition: isDragging ? "none" : "transform 0.1s ease-out",
          }}
          className="relative w-full h-full flex items-center justify-center"
        >
          {/* Loaded Image */}
          <img
            ref={imgRef}
            src={imageUrl}
            alt="High-resolution aerial"
            onLoad={updateImageLayout}
            className="max-w-full max-h-full object-contain pointer-events-none"
            style={{ userSelect: "none" }}
          />

          {/* Cascade stages overlay boxes */}
          {showStages && stages && imgLayout.width > 0 && stages.map((stage) => {
            const cw = stage.crop_window;
            const leftN = cw.left / origW;
            const topN = cw.top / origH;
            const widthN = cw.width / origW;
            const heightN = cw.height / origH;

            // Compute overlays coordinates inside the current img bounds
            const overlayLeft = imgLayout.left + leftN * imgLayout.width;
            const overlayTop = imgLayout.top + topN * imgLayout.height;
            const overlayWidth = widthN * imgLayout.width;
            const overlayHeight = heightN * imgLayout.height;

            const colorClass = stageColors[stage.stage % stageColors.length];

            return (
              <div
                key={stage.stage}
                style={{
                  position: "absolute",
                  left: `${overlayLeft}px`,
                  top: `${overlayTop}px`,
                  width: `${overlayWidth}px`,
                  height: `${overlayHeight}px`,
                  pointerEvents: "none",
                }}
                className={`border border-dashed ${colorClass} rounded-md transition-all duration-300`}
              >
                <div className="absolute top-1 left-1 px-1.5 py-0.5 rounded text-[8px] font-bold leading-none bg-black/70 text-white uppercase tracking-wider backdrop-blur-sm">
                  Stage {stage.stage}: {stage.label}
                </div>
              </div>
            );
          })}

          {/* GCP Keypoint Target Marker */}
          {imgLayout.width > 0 && (
            <div
              style={{
                position: "absolute",
                left: `${markerLeft}px`,
                top: `${markerTop}px`,
                transform: `translate(-50%, -50%) scale(${1 / Math.sqrt(scale)})`, // Scale down so target doesn't get massive when zoomed in
                transformOrigin: "center center",
                pointerEvents: "auto",
              }}
              className="z-10 cursor-pointer"
            >
              <Tooltip>
                <TooltipTrigger
                  render={
                    <div className="relative flex items-center justify-center h-10 w-10 group">
                      {/* Ring ping animations */}
                      <div className="absolute h-8 w-8 rounded-full border-2 border-emerald-500/30 animate-ping" />
                      <div className="absolute h-6 w-6 rounded-full border-2 border-emerald-500/60 scale-95" />
                      
                      {/* Main Target Crosshair */}
                      <Target className="h-5 w-5 text-emerald-400 drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]" />
                      
                      {/* Center point dot */}
                      <div className="absolute h-1.5 w-1.5 rounded-full bg-emerald-300" />
                    </div>
                  }
                />
                <TooltipContent side="top" className="bg-slate-900 border-slate-800 text-slate-100 p-2.5 rounded-xl shadow-lg font-sans">
                  <div className="flex flex-col gap-1 text-[11px]">
                    <div className="flex items-center gap-1.5">
                      <Target className="h-3 w-3 text-emerald-400" />
                      <span className="font-bold text-white">GCP Predicted Pose</span>
                    </div>
                    <div className="border-t border-slate-800 my-1" />
                    <div>Shape: <span className="font-bold text-emerald-400">{shape}</span> ({(confidence * 100).toFixed(1)}%)</div>
                    <div>X: <span className="font-mono font-bold text-slate-300">{x.toFixed(2)} px</span></div>
                    <div>Y: <span className="font-mono font-bold text-slate-300">{y.toFixed(2)} px</span></div>
                  </div>
                </TooltipContent>
              </Tooltip>
            </div>
          )}
        </div>

        {/* Floating Zoom / Pan Controls Overlay */}
        <div className="absolute bottom-4 right-4 flex items-center gap-1.5 bg-background/80 backdrop-blur-md border border-border/60 p-1 rounded-xl shadow-md z-20">
          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={zoomIn}
                  className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground"
                >
                  <ZoomIn className="h-4 w-4" />
                </Button>
              }
            />
            <TooltipContent side="top">Zoom In</TooltipContent>
          </Tooltip>

          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={zoomOut}
                  className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground"
                >
                  <ZoomOut className="h-4 w-4" />
                </Button>
              }
            />
            <TooltipContent side="top">Zoom Out</TooltipContent>
          </Tooltip>

          <div className="w-[1px] h-4 bg-border/80" />

          <Tooltip>
            <TooltipTrigger
              render={
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={resetZoom}
                  className="h-8 w-8 rounded-lg text-muted-foreground hover:text-foreground"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              }
            />
            <TooltipContent side="top">Reset View</TooltipContent>
          </Tooltip>
        </div>

        {/* Small scale badge overlay */}
        <div className="absolute top-4 left-4 bg-black/60 border border-slate-800 text-[10px] font-mono text-slate-300 px-2 py-0.5 rounded-lg backdrop-blur-sm z-20 select-none">
          Zoom: {scale.toFixed(1)}x
        </div>
      </div>
    </div>
  );
}
