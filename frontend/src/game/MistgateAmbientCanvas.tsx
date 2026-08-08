import { Application, Container, Graphics } from "pixi.js";
import { useEffect, useRef } from "react";

export function MistgateAmbientCanvas() {
  const hostRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const app = new Application();
    let cancelled = false;
    let initialized = false;
    let mountedCanvas: HTMLCanvasElement | null = null;
    let mist: Container | null = null;

    void app
      .init({ resizeTo: host, backgroundAlpha: 0, antialias: true, resolution: devicePixelRatio })
      .then(() => {
        initialized = true;
        if (cancelled) {
          app.destroy(true);
          return;
        }
        mountedCanvas = app.canvas;
        host.appendChild(mountedCanvas);

        const skyline = new Graphics()
          .poly([0, 340, 0, 225, 88, 190, 142, 226, 214, 145, 286, 220, 360, 175, 445, 228, 540, 166, 640, 218, 760, 188, 880, 235, 960, 198, 960, 340])
          .fill({ color: 0x143b46, alpha: 0.42 });
        skyline.scale.set(app.screen.width / 960, app.screen.height / 340);
        app.stage.addChild(skyline);

        mist = new Container();
        for (let index = 0; index < 7; index += 1) {
          const cloud = new Graphics()
            .ellipse(110 + index * 148, 175 + (index % 2) * 42, 180, 54)
            .fill({ color: index % 2 ? 0xc9f7ec : 0xdbeafc, alpha: 0.13 });
          mist.addChild(cloud);
        }
        app.stage.addChild(mist);

        const lantern = new Graphics()
          .circle(app.screen.width * 0.76, app.screen.height * 0.27, 5)
          .fill({ color: 0xffd892, alpha: 0.92 })
          .circle(app.screen.width * 0.76, app.screen.height * 0.27, 24)
          .fill({ color: 0xffd892, alpha: 0.08 });
        app.stage.addChild(lantern);

        const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        if (!reduceMotion) {
          app.ticker.add(() => {
            if (mist) mist.x = Math.sin(app.ticker.lastTime / 4200) * 24;
          });
        }
      });

    return () => {
      cancelled = true;
      if (mountedCanvas?.parentElement === host) host.removeChild(mountedCanvas);
      if (initialized) app.destroy(true, { children: true });
    };
  }, []);

  return <div ref={hostRef} className="ambient-canvas" aria-hidden="true" />;
}
