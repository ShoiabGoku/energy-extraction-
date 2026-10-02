# Energy extraction from electric eels, used for the right purposes

A machine design with **real 3D models** that open in any 3D program, plus the engineering behind them.

![Passive eel habitat harvester](docs/img/habitat.png)

An electric eel can fire up to **860 V**, but each pulse lasts only about **2 ms**. Wall electrodes in its tank collect about **0.1 J a day**, so one eel would need about a thousand years to charge a phone. Taking more would mean provoking the animal, and that is cruel. The design therefore has two parts:

| | Part A: passive habitat harvester | Part B: artificial electric organ |
|---|---|---|
| Idea | Collect only the pulses the eel fires anyway | Copy the eel's cells in hydrogel instead of using the eel |
| Output | Microwatts, event-driven | Volts from salt gradients, scalable |
| Used for | Keeper shock alarm, eel health log, education display | Implants without battery-replacement surgery, soft robots, blue energy |
| Model | [`models/eel-habitat-harvester.*`](models) | [`models/artificial-electric-organ.*`](models) |

Read the full design, with the energy derivations, electronics, bill of materials, welfare rules and references, in **[docs/DESIGN.md](docs/DESIGN.md)**.

![System diagram](docs/system-diagram.svg)

## 3D models

Each model is exported in three formats so that anything can open it:

| File | Format | Opens in | Notes |
|---|---|---|---|
| `*.glb` | glTF 2.0 binary | Blender, Unity, Unreal, Godot, three.js / Babylon.js, Windows 3D Viewer, KeyShot, online glTF viewers | Full PBR materials, named part hierarchy, a description on every part (`extras.description`). Passes the Khronos glTF Validator with 0 errors and 0 warnings |
| `*.obj` + `*.mtl` | Wavefront OBJ | Practically every 3D program: Maya, 3ds Max, Cinema 4D, SketchUp, Rhino, MeshLab… | One named object per part with colours, emission and opacity |
| `*.stl` | Binary STL | Slicers (Cura, PrusaSlicer), CAD (Fusion, SolidWorks, FreeCAD), and **GitHub's built-in 3D preview** (click the file) | Single colour, **millimetres, +Z up**, opaque parts only (glass, water, field lines and clear lids left out so you can see inside) |

GLB and OBJ use **metres, +Y up**. The habitat model is 3.7 × 1.9 × 1.9 m including the pedestal and display. The organ model is 380 × 200 × 87 mm.

**What's inside the models**

- **Habitat harvester** (44 parts, 81k triangles):
  - the tank, water, hide tube, rocks and lid with feeding hatch
  - a 1.7 m electric eel with its long anal fin
  - the eel's head-to-tail discharge field lines (illustrative)
  - six graphite plates behind acrylic guard screens, with their leads
  - the harvester enclosure with a see-through lid: 12-diode polyphase bridge, surge clamp, catch capacitor, converter and supercapacitors
  - the three loads: health monitor and probe, alarm beacon, e-paper display
- **Artificial electric organ** (23 parts, 22k triangles):
  - a 20-cell hydrogel stack in a silicone sleeve, with Ag/AgCl electrodes
  - leads to an implant
  - an exploded single cell with Na⁺ / Cl⁻ ion-flow arrows and the net current arrow

| Habitat harvester | Artificial electric organ |
|---|---|
| ![Eel and discharge field](docs/img/eel-field.png) | ![Organ](docs/img/organ.png) |
| ![Harvester electronics](docs/img/harvester.png) | Every part carries its own description. Hover over it in the viewer, or check the custom properties in Blender. |

### Quick start

- **Blender:** File → Import → glTF 2.0 → `models/eel-habitat-harvester.glb`. The water and glass are transparent; the field lines and alarm beacon glow.
- **Windows:** double-click the `.glb` file to open it in 3D Viewer.
- **Browser:** open `index.html` through any static server (for example `python -m http.server`), or enable GitHub Pages for this repo. The viewer has layer toggles, hover-for-description, see-through mode, auto-rotate and PNG export.
- **Print or CAD:** use the `.stl` file.

## Regenerating the models

The models are generated from code, so every dimension in [`tools/build_models.py`](tools/build_models.py) can be changed and rebuilt:

```bash
pip install numpy
python tools/build_models.py
```

The output is deterministic, so the same code always produces the same files.

## The rules built into this machine

1. **Passive only.** No prodding, baiting, stimulation, or feeding schedules changed to make the eel fire more.
2. **No added effort for the eel.** The harvester may add at most 5% to the eel's load; the design adds about 0.1%.
3. **Welfare comes before output.** The health monitor runs on its own battery, so it never goes dark when a sick eel discharges less.
4. **No surgery, implants or restraint.** Only captive-bred or rescued eels in accredited facilities.
5. **No weapons or stunning devices.** The 3.3 V-only output and small energy store make that a hardware limit.

Energy figures are order-of-magnitude estimates with their assumptions stated in [DESIGN.md](docs/DESIGN.md). They should be refined with measurements from a real habitat.
