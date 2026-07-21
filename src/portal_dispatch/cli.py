"""Command-line entrypoints."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="portal-dispatch")
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="Train routers from a precomputed score bundle and sweep policies.")
    smoke.add_argument("--bundle", required=True, help="Path to scores bundle (joblib).")
    smoke.add_argument("--out", default="results.json")
    serve = sub.add_parser("serve", help="Run the production gateway from a routes.yaml config.")
    serve.add_argument("--config", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--traces", default="traces.jsonl")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn

        from .serve import GatewayConfig, create_production_app
        from .tracing import TraceJournal

        app = create_production_app(GatewayConfig.from_yaml(args.config),
                                    journal=TraceJournal(args.traces))
        uvicorn.run(app, host=args.host, port=args.port)
        return

    if args.command == "smoke":
        from .sweep import run_policy_sweep

        report = run_policy_sweep(args.bundle, args.out)
        for stats in report["_stats"]:
            print(
                f"{stats.name:32s} acc={stats.accuracy:.3f} savings={stats.savings:.3f} "
                f"drop={stats.drop_vs_capable:+.3f}"
            )
        print("kill_criteria:", report["kill_criteria"])


if __name__ == "__main__":
    main()
