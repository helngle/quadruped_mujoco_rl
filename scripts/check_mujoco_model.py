from pathlib import Path

import mujoco


def main() -> None:
    model_path = Path("assets/robots/quadruped.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    for _ in range(10):
        mujoco.mj_step(model, data)

    print(f"Loaded {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}, bodies={model.nbody}")


if __name__ == "__main__":
    main()
