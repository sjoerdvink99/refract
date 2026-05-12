import { useEffect } from "react";
import { useSelectionStore } from "@/state/selectionStore";

export function useKeyboardShortcuts() {
  const { clearSelection } = useSelectionStore();

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        clearSelection();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [clearSelection]);
}
