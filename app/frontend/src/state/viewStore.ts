import { create } from "zustand";
import type { ColorMode } from "@/api/types";

interface ViewState {
  colorBy: ColorMode;
  egoRadius: 1 | 2;
  statsOpen: boolean;
  setColorBy: (mode: ColorMode) => void;
  setEgoRadius: (r: 1 | 2) => void;
  setStatsOpen: (v: boolean) => void;
}

export const useViewStore = create<ViewState>((set) => ({
  colorBy: "regime",
  egoRadius: 1,
  statsOpen: false,
  setColorBy: (colorBy) => set({ colorBy }),
  setEgoRadius: (egoRadius) => set({ egoRadius }),
  setStatsOpen: (statsOpen) => set({ statsOpen }),
}));
