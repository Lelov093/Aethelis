import { Application, Assets, Container, Graphics, Sprite, Text, type Texture } from "pixi.js";
import { useEffect, useRef } from "react";

import type { SceneView } from "../api/contracts";

const WORLD_WIDTH = 1680;
const WORLD_HEIGHT = 980;
const INTERACTION_RADIUS = 150;
const WALK_SPEED = 330;

const ENTITY_SLOTS: readonly [number, number][] = [
  [1110, 520],
  [610, 455],
  [1310, 670],
  [420, 700],
  [895, 335],
  [1450, 410],
];

const RESOURCE_SLOTS: readonly [number, number][] = [
  [760, 690],
  [1225, 755],
  [500, 570],
  [980, 790],
];

interface MistgateWorldCanvasProps {
  scene: SceneView;
  backgroundUrl?: string;
  onNearbyTargetChange: (targetId: string | null) => void;
  onInteract: (targetId: string | null) => void;
}

interface WorldPoint {
  id: string;
  x: number;
  y: number;
}

export function MistgateWorldCanvas({
  scene,
  backgroundUrl,
  onNearbyTargetChange,
  onInteract,
}: MistgateWorldCanvasProps) {
  const canvasHostRef = useRef<HTMLDivElement>(null);
  const nearbyCallbackRef = useRef(onNearbyTargetChange);
  const interactCallbackRef = useRef(onInteract);

  nearbyCallbackRef.current = onNearbyTargetChange;
  interactCallbackRef.current = onInteract;

  useEffect(() => {
    const host = canvasHostRef.current;
    if (!host) return;
    if (navigator.userAgent.includes("jsdom")) return;

    const app = new Application();
    const pressed = new Set<string>();
    let cancelled = false;
    let mountedCanvas: HTMLCanvasElement | null = null;
    let pointerDownHandler: ((event: PointerEvent) => void) | null = null;
    let destination: { x: number; y: number } | null = null;
    let nearbyId: string | null = null;

    const player = { x: 840, y: 680 };
    host.dataset.playerPosition = `${player.x},${player.y}`;
    const points: WorldPoint[] = [
      ...scene.visible_entities.map((entity, index) => {
        const [x, y] = ENTITY_SLOTS[index % ENTITY_SLOTS.length];
        return { id: entity.id, x, y };
      }),
      ...scene.visible_resources.map((resource, index) => {
        const [x, y] = RESOURCE_SLOTS[index % RESOURCE_SLOTS.length];
        return { id: resource.id, x, y };
      }),
    ];

    const keyDown = (event: KeyboardEvent) => {
      if (isTextEntry(event.target)) return;
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) {
        pressed.add(event.code);
        destination = null;
        event.preventDefault();
      }
      if ((event.code === "KeyE" || event.code === "Enter") && !event.repeat) {
        interactCallbackRef.current(nearbyId);
        event.preventDefault();
      }
    };
    const keyUp = (event: KeyboardEvent) => pressed.delete(event.code);

    window.addEventListener("keydown", keyDown);
    window.addEventListener("keyup", keyUp);

    void app
      .init({
        resizeTo: host,
        background: 0x0b252c,
        antialias: true,
        autoDensity: true,
        resolution: Math.min(window.devicePixelRatio || 1, 2),
      })
      .then(async () => {
        if (cancelled) {
          app.destroy(true);
          return;
        }
        mountedCanvas = app.canvas;
        mountedCanvas.setAttribute("aria-hidden", "true");
        host.appendChild(mountedCanvas);

        const world = new Container();
        app.stage.addChild(world);

        if (backgroundUrl) {
          try {
            const texture = await Assets.load<Texture>(backgroundUrl);
            if (!cancelled) {
              const background = new Sprite(texture);
              background.width = WORLD_WIDTH;
              background.height = WORLD_HEIGHT;
              background.alpha = 0.9;
              world.addChild(background);
            }
          } catch {
            host.dataset.backgroundFallback = "true";
          }
        }

        const atmosphere = new Graphics()
          .rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
          .fill({ color: 0x0a2930, alpha: 0.2 })
          .ellipse(840, 790, 780, 225)
          .fill({ color: 0xd9f7ee, alpha: 0.08 });
        world.addChild(atmosphere);

        const walkable = new Graphics()
          .ellipse(840, 735, 700, 205)
          .fill({ color: 0x9dd8c8, alpha: 0.07 })
          .stroke({ color: 0xdaf8ee, alpha: 0.12, width: 3 });
        world.addChild(walkable);

        for (const [index, entity] of scene.visible_entities.entries()) {
          const point = points.find((item) => item.id === entity.id)!;
          world.addChild(characterMarker(entity.name, point.x, point.y, index, () => {
            destination = approachPoint(player, point);
          }));
        }

        for (const resource of scene.visible_resources) {
          const point = points.find((item) => item.id === resource.id)!;
          world.addChild(resourceMarker(resource.name, point.x, point.y, () => {
            destination = approachPoint(player, point);
          }));
        }

        const playerView = playerMarker();
        world.addChild(playerView);

        pointerDownHandler = (event: PointerEvent) => {
          const bounds = mountedCanvas!.getBoundingClientRect();
          const scale = world.scale.x || 1;
          destination = {
            x: clamp((event.clientX - bounds.left - world.x) / scale, 90, WORLD_WIDTH - 90),
            y: clamp((event.clientY - bounds.top - world.y) / scale, 320, WORLD_HEIGHT - 95),
          };
          host.focus();
        };
        mountedCanvas.addEventListener("pointerdown", pointerDownHandler);

        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        app.ticker.add((ticker) => {
          const seconds = Math.min(ticker.deltaMS / 1000, 0.05);
          let dx = Number(pressed.has("ArrowRight") || pressed.has("KeyD")) - Number(pressed.has("ArrowLeft") || pressed.has("KeyA"));
          let dy = Number(pressed.has("ArrowDown") || pressed.has("KeyS")) - Number(pressed.has("ArrowUp") || pressed.has("KeyW"));

          if (!dx && !dy && destination) {
            dx = destination.x - player.x;
            dy = destination.y - player.y;
            if (Math.hypot(dx, dy) < 12) destination = null;
          }

          const distance = Math.hypot(dx, dy);
          if (distance > 0) {
            player.x = clamp(player.x + (dx / distance) * WALK_SPEED * seconds, 90, WORLD_WIDTH - 90);
            player.y = clamp(player.y + (dy / distance) * WALK_SPEED * seconds, 320, WORLD_HEIGHT - 95);
            host.dataset.playerPosition = `${Math.round(player.x)},${Math.round(player.y)}`;
            playerView.scale.x = dx < 0 ? -1 : 1;
          }
          playerView.position.set(player.x, player.y + (reduceMotion ? 0 : Math.sin(app.ticker.lastTime / 180) * Math.min(distance, 1) * 3));

          const nearest = points
            .map((point) => ({ ...point, distance: Math.hypot(point.x - player.x, point.y - player.y) }))
            .sort((left, right) => left.distance - right.distance)[0];
          const nextNearby = nearest && nearest.distance <= INTERACTION_RADIUS ? nearest.id : null;
          if (nextNearby !== nearbyId) {
            nearbyId = nextNearby;
            nearbyCallbackRef.current(nearbyId);
          }

          const scale = Math.max(app.screen.width / 1320, app.screen.height / 760);
          world.scale.set(scale);
          const targetX = app.screen.width / 2 - player.x * scale;
          const targetY = app.screen.height * 0.6 - player.y * scale;
          world.x += (targetX - world.x) * (reduceMotion ? 1 : 0.1);
          world.y += (targetY - world.y) * (reduceMotion ? 1 : 0.1);
        });
      })
      .catch(() => {
        host.dataset.renderingFallback = "true";
      });

    return () => {
      cancelled = true;
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("keyup", keyUp);
      nearbyCallbackRef.current(null);
      if (mountedCanvas && pointerDownHandler) mountedCanvas.removeEventListener("pointerdown", pointerDownHandler);
      if (mountedCanvas?.parentElement === host) host.removeChild(mountedCanvas);
      try {
        app.destroy(true, { children: true });
      } catch {
        // Initialization may have failed before Pixi created a renderer.
      }
    };
  }, [backgroundUrl, scene.location_id, scene.visible_entities, scene.visible_resources]);

  return (
    <div className="world-canvas-frame">
      <div
        ref={canvasHostRef}
        className="world-canvas-host"
        role="application"
        tabIndex={0}
        aria-label={`${scene.location_name ?? "Mistgate"} 可移动游戏场景。使用方向键或 WASD 移动，靠近人物或物件后按 E 互动。`}
      />
      <div className="world-control-legend" aria-hidden="true">
        <span><kbd>WASD</kbd> 移动</span><span><kbd>E</kbd> 互动</span><span>点击地面前往</span>
      </div>
      <div className="sr-only" aria-live="polite">
        当前场景人物：{scene.visible_entities.map((item) => item.name).join("、") || "无"}
      </div>
    </div>
  );
}

function isTextEntry(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (
    target.isContentEditable || target.matches("input, textarea, select, [role='textbox']")
  );
}

function playerMarker(): Container {
  const marker = new Container();
  const shadow = new Graphics().ellipse(0, 27, 34, 11).fill({ color: 0x061a20, alpha: 0.38 });
  const body = new Graphics()
    .circle(0, -4, 25)
    .fill({ color: 0xdff8ef })
    .stroke({ color: 0xffffff, alpha: 0.92, width: 4 })
    .poly([-18, 15, 0, -2, 18, 15, 13, 37, -13, 37])
    .fill({ color: 0x258f88 });
  const sigil = new Text({ text: "Æ", style: { fill: 0x164c51, fontSize: 18, fontWeight: "700" } });
  sigil.anchor.set(0.5);
  sigil.y = -5;
  marker.addChild(shadow, body, sigil);
  marker.position.set(840, 680);
  return marker;
}

function characterMarker(name: string, x: number, y: number, index: number, onSelect: () => void): Container {
  const marker = new Container();
  marker.position.set(x, y);
  marker.eventMode = "static";
  marker.cursor = "pointer";
  marker.on("pointertap", onSelect);
  const colors = [0xc49578, 0x8f7caf, 0x507f8b, 0x9b6f78];
  marker.addChild(
    new Graphics().ellipse(0, 28, 31, 10).fill({ color: 0x061a20, alpha: 0.34 }),
    new Graphics()
      .circle(0, -3, 23)
      .fill({ color: 0xffedcb })
      .stroke({ color: 0xffffff, alpha: 0.72, width: 3 })
      .poly([-17, 14, 0, 0, 17, 14, 12, 35, -12, 35])
      .fill({ color: colors[index % colors.length] }),
  );
  const label = new Text({ text: name, style: { fill: 0xffffff, fontSize: 17, fontWeight: "600", stroke: { color: 0x08242a, width: 4 } } });
  label.anchor.set(0.5, 0);
  label.y = 45;
  marker.addChild(label);
  return marker;
}

function resourceMarker(name: string, x: number, y: number, onSelect: () => void): Container {
  const marker = new Container();
  marker.position.set(x, y);
  marker.eventMode = "static";
  marker.cursor = "pointer";
  marker.on("pointertap", onSelect);
  marker.addChild(
    new Graphics().ellipse(0, 18, 28, 9).fill({ color: 0x061a20, alpha: 0.3 }),
    new Graphics().roundRect(-21, -21, 42, 42, 10).fill({ color: 0xe9c77c }).stroke({ color: 0xfff6d5, width: 3 }),
  );
  const label = new Text({ text: name, style: { fill: 0xffffff, fontSize: 15, fontWeight: "600", stroke: { color: 0x08242a, width: 4 } } });
  label.anchor.set(0.5, 0);
  label.y = 31;
  marker.addChild(label);
  return marker;
}

function approachPoint(player: { x: number; y: number }, target: WorldPoint) {
  const dx = player.x - target.x;
  const dy = player.y - target.y;
  const distance = Math.max(Math.hypot(dx, dy), 1);
  return { x: target.x + (dx / distance) * 95, y: target.y + (dy / distance) * 95 };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
