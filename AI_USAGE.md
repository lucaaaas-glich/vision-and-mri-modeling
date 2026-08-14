# AI Usage Record

## Purpose and scope

ChatGPT was used as a learning assistant during COMP3710 Lab 1. Its role was to
explain PyTorch concepts, help break the lab into stages, suggest implementation
approaches, and review the resulting code. I remained responsible for running
the programs, checking their output, modifying the code for my Mac, and
preparing to explain the algorithms during the demonstration.

This record distinguishes preserved chat text from reconstructed summaries.
Some early messages are no longer available word for word. The reconstructed
items below describe the intent of those interactions and are not presented as
exact quotations.

## Preserved chat record

The following is a preserved part of the ChatGPT response that established the
learning plan:

> 我建议我们怎么学（也是整个学期效率最高的方法）
>
> 你之前希望我当你的 **COMP3710 老师**，而不是简单回答问题。我建议我们不要直接把代码复制进去，而是按实验节奏完成。
>
> 第一步（今天）
>
> 完成 Part 1。
>
> 我会带你：
>
> 为什么要用 Tensor
>
> 为什么不用 NumPy
>
> 每一行代码什么意思
>
> 最后完成 Gaussian、2D Sine 和 Gabor Filter
>
> 这一部分也是后面深度学习最重要的基础。
>
> 第二步
>
> 完成 Part 2。
>
> 实现 Mandelbrot，再改成 Julia，并教你理解复数、迭代和并行计算，而不是只会改代码。
>
> 第三步
>
> 帮你做 Part 3。
>
> 我会推荐一个**既容易拿高分，又能体现 PyTorch 并行计算**的 Fractal，并指导你完成 GitHub 提交、AI Prompt 记录和展示准备。

This response proposed a staged teaching workflow instead of a single prompt
that generated the final project.

## Reconstructed interaction summary

The original wording of every early prompt is not available. Based on the
preserved chat and the resulting files, the interactions had the following
goals.

### Part 1: Gaussian, sine, and Gabor filter

Reconstructed prompt intent:

> Teach me how to complete Part 1 step by step. Explain why PyTorch tensors are
> used, explain each line, generate a 2D Gaussian and sine function, and show
> their product as a Gabor filter.

AI assistance:

- explained the conversion from NumPy coordinate grids to PyTorch tensors;
- explained moving tensors to an available device;
- suggested computing a two-dimensional sine wave from the x and y grids; and
- explained that multiplying the Gaussian envelope by the sine wave produces a
  Gabor filter.

Student work and verification:

- adapted device selection for an Apple Silicon Mac by using MPS with a CPU
  fallback;
- ran the program in PyCharm;
- checked the PyTorch version and selected device printed by the program; and
- visually checked that the output was a sinusoidal pattern modulated by a
  Gaussian envelope.

### Part 2: Mandelbrot zoom and Julia set

Reconstructed prompt intent:

> Teach me how the Mandelbrot iteration works in PyTorch, then help me increase
> the resolution, zoom into another region, and modify it into a Julia set.

AI assistance:

- explained the iteration `z = z * z + c` and the escape test;
- suggested using tensor operations over the complete complex-plane grid;
- explained how reducing the grid spacing increases image resolution; and
- explained the key difference between Mandelbrot and Julia initial conditions.

Student work and verification:

- used an MPS/CPU device choice appropriate for a Mac rather than copying the
  CUDA-only configuration from the lab sheet;
- changed the zoom window to real values from -1.0 to -0.5 and imaginary values
  from -0.25 to 0.25;
- reduced the grid spacing to 0.0005 and increased the iteration count to 500;
- created a Julia set with the fixed constant `c = -0.8 + 0.156i`; and
- ran and visually inspected the Mandelbrot, zoom, and Julia outputs.

### Part 3: Barnsley Fern

Reconstructed prompt intent:

> Recommend a fractal that is clearly different from the Mandelbrot set and can
> demonstrate meaningful PyTorch parallel computation. Teach me how to implement
> it and prepare it for a GitHub demonstration.

AI assistance:

- suggested the Barnsley Fern as an Iterated Function System;
- explained the four affine transformations and their probabilities;
- suggested using many walkers so that PyTorch can update many points in
  parallel; and
- helped organise the repository, documentation, and GitHub push workflow.

Student work and verification:

- used 20,000 walkers and 150 iterations, with the first 20 iterations removed
  as burn-in;
- used tensor random-number generation, Boolean masks, and indexed tensor
  assignments for the four transformations;
- selected MPS on Apple Silicon with a CPU fallback;
- saved and inspected the high-resolution fern image; and
- created a GitHub repository and verified that the local and remote `main`
  branches pointed to the same commit.

## Limitations and issues identified

AI-generated or AI-assisted code still required review. Important issues found
during review included:

- generic examples often assume CUDA, which is not the correct accelerator for
  an Apple Silicon Mac;
- the first repository version had a file named ` .gitignore` with a leading
  space, so Git did not recognise it as an ignore file;
- the first README was too brief for an assessment demonstration and contained
  an unclosed Markdown code block;
- the first Part 1 script displayed only the final Gabor result even though the
  Gaussian and sine tensors were also calculated;
- transferring fern points from MPS to the CPU during every iteration is easy
  to understand but is not the most efficient parallel implementation; and
- a single generated image is not enough to demonstrate substantial analysis
  of the fractal.

These limitations are being addressed through separate, reviewable Git commits.

## Further analysis and academic responsibility

The final project will include analysis beyond producing one image, including a
fractal-dimension estimate and a comparison of computational settings. The
Barnsley Fern choice must also be confirmed with the teaching team because the
lab specification requires approval for a fractal selected from the referenced
Wikipedia list.

I will not claim that reconstructed prompts are verbatim. If additional chat
history becomes available, it will be added to this record before the lab
demonstration.
