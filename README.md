# PyTorch Barnsley Fern

This project generates a Barnsley Fern fractal with an Iterated Function
System (IFS). PyTorch tensors are used to update 20,000 independent walkers in
parallel. On Apple Silicon, the program uses the MPS device when available and
falls back to the CPU otherwise.

![Barnsley Fern](images/barnsley_fern.png)

## How the fractal is formed

Each point `(x, y)` is transformed by one of four affine transformations. The
transformation is selected randomly using the probabilities below.

| Part of fern | Probability | New x | New y |
| --- | ---: | --- | --- |
| Stem | 1% | `0` | `0.16y` |
| Main leaflets | 85% | `0.85x + 0.04y` | `-0.04x + 0.85y + 1.6` |
| Left leaflet | 7% | `0.20x - 0.26y` | `0.23x + 0.22y + 1.6` |
| Right leaflet | 7% | `-0.15x + 0.28y` | `0.26x + 0.24y + 0.44` |

Repeated application of these transformations produces the self-similar fern
structure. The first 20 iterations are discarded to reduce the influence of
the initial point at the origin.

## How PyTorch is used

The core fractal computation is tensorised. At every iteration, PyTorch:

1. generates one random value for every walker;
2. creates Boolean masks that select the four transformations;
3. updates all selected points in parallel; and
4. runs the calculation on MPS when supported, otherwise on the CPU.

This makes PyTorch a major component of the fractal algorithm rather than using
it only for storage or plotting.

## Requirements

- Python 3.11 or a compatible Python 3 version
- PyTorch
- Matplotlib
- NumPy

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Run

From the repository directory, run:

```bash
python barnsley_fern.py
```

The program displays the fractal and saves a high-resolution image to
`images/barnsley_fern.png`.

## Project structure

```text
.
|-- barnsley_fern.py
|-- images/
|   `-- barnsley_fern.png
|-- requirements.txt
`-- README.md
```

## AI-assisted development

AI tools were used during development. The prompts, suggestions, verification
steps, limitations, and student changes will be recorded in `AI_USAGE.md`.

## Course approval

The lab specification asks students to seek teaching-team approval for a
fractal selected from the referenced Wikipedia list. Approval of the Barnsley
Fern should be confirmed with the demonstrator before assessment.
