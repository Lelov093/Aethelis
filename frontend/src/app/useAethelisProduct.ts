import { useCallback, useEffect, useState } from "react";

import { aethelisApi, AethelisApiError } from "../api/client";
import type {
  AvailableWorldContent,
  PlayerProfile,
  PlaySession,
  SavePointView,
  WorldTimelineView,
} from "../api/contracts";

export interface ProductState {
  profile: PlayerProfile | null;
  content: AvailableWorldContent[];
  timelines: WorldTimelineView[];
  archivedTimelines: WorldTimelineView[];
  loading: boolean;
  error: string | null;
}

const initialState: ProductState = {
  profile: null,
  content: [],
  timelines: [],
  archivedTimelines: [],
  loading: true,
  error: null,
};

function errorMessage(error: unknown): string {
  return error instanceof AethelisApiError ? error.problem.detail : "发生了未预期的本地错误。";
}

export function useAethelisProduct() {
  const [state, setState] = useState<ProductState>(initialState);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setState((current) => ({ ...current, loading: true, error: null }));
    try {
      const [profile, content, timelines, allTimelines] = await Promise.all([
        aethelisApi.getProfile(),
        aethelisApi.listWorldContent(),
        aethelisApi.listTimelines(),
        aethelisApi.listTimelines(true),
      ]);
      setState({
        profile,
        content,
        timelines,
        archivedTimelines: allTimelines.filter((timeline) => timeline.status === "archived"),
        loading: false,
        error: null,
      });
    } catch (error) {
      setState((current) => ({ ...current, loading: false, error: errorMessage(error) }));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const run = useCallback(async <T,>(label: string, operation: () => Promise<T>): Promise<T> => {
    setBusyAction(label);
    try {
      const result = await operation();
      await refresh();
      return result;
    } catch (error) {
      setState((current) => ({ ...current, error: errorMessage(error) }));
      throw error;
    } finally {
      setBusyAction(null);
    }
  }, [refresh]);

  return {
    state,
    busyAction,
    refresh,
    createTimeline: (contentVersionId: string, name: string) => {
      if (!state.profile) throw new Error("local profile is not loaded");
      return run("create", () =>
        aethelisApi.createTimeline({
          content_version_id: contentVersionId,
          player_profile_id: state.profile!.id,
          name,
        }),
      );
    },
    startSession: (worldId: string): Promise<PlaySession> => {
      if (!state.profile) throw new Error("local profile is not loaded");
      return run("continue", () => aethelisApi.startSession(worldId, state.profile!.id));
    },
    listSaves: (worldId: string): Promise<SavePointView[]> => aethelisApi.listSaves(worldId),
    createSave: (worldId: string, name: string) =>
      run(`save:${worldId}`, () => aethelisApi.createSave(worldId, name)),
    forkSave: (worldId: string, saveId: string, name: string) => {
      if (!state.profile) throw new Error("local profile is not loaded");
      return run(`fork:${saveId}`, () =>
        aethelisApi.forkSave(worldId, saveId, state.profile!.id, name),
      );
    },
    archiveTimeline: (worldId: string) =>
      run(`archive:${worldId}`, () => aethelisApi.archiveTimeline(worldId)),
    restoreTimeline: (worldId: string) =>
      run(`restore:${worldId}`, () => aethelisApi.restoreTimeline(worldId)),
  };
}
