"""续训约束与原子写盘：防止不同算法/数据/源码混用，以及checkpoint与日志错一轮。"""

from pathlib import Path
import hashlib
import json
import os
import pickle


def run_signature(args, root):
    paths = [
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "datasets/dextrack").glob("*.npz")),
        root / "assets/allegro_plastic_cube.xml",
    ]
    return dict(
        schema=1,
        num_envs=getattr(args, "num_envs", 2),
        pods=args.pods,
        horizon=args.horizon,
        feasible_only=args.feasible_only,
        velocity_feedforward=args.velocity_feedforward,
        feasibility_sha256=(
            hashlib.sha256(Path(args.feasibility_report).read_bytes()).hexdigest()
            if args.feasibility_report
            else None
        ),
        files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )


def validate_run_directory(output, signature, *, resume):
    signature_path = output / "run_signature.json"
    checkpoint_path = output / "checkpoint_latest.pkl"
    if resume:
        if not signature_path.exists() or not checkpoint_path.exists():
            raise ValueError("缺少带配置指纹的checkpoint；旧验证文件不能直接作为严格续训起点")
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError("续训配置、数据或源码与原运行不一致；请恢复原设置或使用新输出目录")
    else:
        if checkpoint_path.exists() or signature_path.exists():
            raise ValueError("输出目录已有训练记录；请使用--resume或新的输出目录")
        output.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2))


def save_latest(path, state, history):
    # 状态与已完成轮数/日志打包为同一个文件，避免两个文件只写成功一个。
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(dict(state=state, history=history), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_latest(path):
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    if not isinstance(saved, dict) or set(saved) != {"state", "history"}:
        raise ValueError("checkpoint格式不匹配")
    if [row["round"] for row in saved["history"]] != list(range(1, len(saved["history"]) + 1)):
        raise ValueError("checkpoint轮数记录不连续")
    return saved["state"], saved["history"]
