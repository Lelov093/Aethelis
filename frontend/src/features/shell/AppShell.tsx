import type { ReactNode } from "react";

export type ProductView = "home" | "new-game" | "play" | "timelines" | "settings";

const navigation: { id: Exclude<ProductView, "play">; label: string; hint: string }[] = [
  { id: "home", label: "返回雾门", hint: "概览" },
  { id: "new-game", label: "开始新旅程", hint: "New Game" },
  { id: "timelines", label: "存档与时间线", hint: "Archives" },
  { id: "settings", label: "设置", hint: "Local" },
];

interface AppShellProps {
  activeView: ProductView;
  displayName: string;
  apiConnected: boolean;
  onNavigate: (view: ProductView) => void;
  children: ReactNode;
}

export function AppShell({ activeView, displayName, apiConnected, onNavigate, children }: AppShellProps) {
  const immersive = activeView === "play";
  return (
    <div className={immersive ? "product-shell immersive-shell" : "product-shell"}>
      {!immersive && <aside className="side-rail" aria-label="产品导航">
        <button className="brand" onClick={() => onNavigate("home")} aria-label="返回 Aethelis 首页">
          <span className="brand-mark">Æ</span>
          <span><strong>Aethelis</strong><small>雾门档案城</small></span>
        </button>
        <nav className="primary-nav">
          {navigation.map((item) => (
            <button
              key={item.id}
              className={activeView === item.id ? "nav-item active" : "nav-item"}
              onClick={() => onNavigate(item.id)}
              aria-current={activeView === item.id ? "page" : undefined}
            >
              <span>{item.label}</span><small>{item.hint}</small>
            </button>
          ))}
        </nav>
        <div className="local-identity">
          <span className={apiConnected ? "status-dot online" : "status-dot"} />
          <span><small>{apiConnected ? "本地世界已连接" : "等待本地世界"}</small><strong>{displayName}</strong></span>
        </div>
      </aside>}
      <main className="main-stage">{children}</main>
      {!immersive && <nav className="mobile-nav" aria-label="移动端产品导航">
        {navigation.map((item) => (
          <button key={item.id} className={activeView === item.id ? "active" : ""} onClick={() => onNavigate(item.id)}>
            {item.label}
          </button>
        ))}
      </nav>}
    </div>
  );
}
