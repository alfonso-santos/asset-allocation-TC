# Project Overview

## Purpose

`tc_synthetic` is a modular Python toolbox for generating synthetic financial return data.
Its goal is to make structural assumptions explicit and testable instead of embedding them in a monolithic generator.

## Problem Addressed

Research on synthetic return generation often mixes several concerns in the same implementation layer:

- scenario definition
- parameter validation
- structural assumptions
- marginal distribution choices
- dependence modeling
- regime logic

That makes it difficult to understand what is being assumed, to test each assumption independently, and to extend the system without side effects.

`tc_synthetic` addresses this by splitting the problem into small modules with narrow responsibilities.
The current codebase focuses on building the foundation for a scenario language and a set of validated structural and univariate generators.

## Current Scope

The project currently implements four core phases.

### 1. Specifications

The `specs` module defines typed configuration objects for marginals, copulas, structures, state processes, and full scenarios.
These specifications validate basic structural correctness and allowed `kind` values.

### 2. Shared Utilities

The `utils` module provides validation helpers and matrix checks used across the toolbox.
This includes:

- random generator creation
- positive integer validation for assets and observations
- square, symmetric, unit-diagonal, and positive semidefinite matrix validation

### 3. Structures

The `structures` module implements structural building blocks for the cross-sectional layer.
Currently this includes:

- equicorrelation matrices
- block correlation matrices
- near-duplicate groups represented as a specialized block model
- one-factor and multi-factor correlation matrices
- explicit nonlinear redundancy groups represented as metadata rather than a fake linear correlation matrix

### 4. Marginals

The `marginals` module implements the univariate side of the toolbox.
Currently this includes:

- empirical standardization for a 1D sample
- standard normal marginal sampling
- standardized Student-t sampling using theoretical variance correction
- standardized skew-normal sampling using theoretical mean and variance correction
- a heterogeneous wrapper that combines multiple `MarginalSpec` objects column by column

## What the Project Does Not Do Yet

The current repository does not yet implement the full end-to-end generator pipeline.
In particular, it does not yet provide:

- multivariate dependence driven by copulas
- copula calibration or sampling logic
- state-process-driven regime switching
- scenario orchestration from `ScenarioSpec` to final datasets
- diagnostics, reporting, or plotting logic beyond placeholder modules
- special generator families beyond the core building blocks already implemented

The repository contains placeholder modules for several of these layers, but they are intentionally empty at the moment.

## Design Position

The project intentionally prefers explicit intermediate representations over premature abstraction.
Examples of this position include:

- keeping `MarginalSpec`, `CopulaSpec`, `StructureSpec`, and related objects as small dataclasses
- representing nonlinear redundancy as groups instead of forcing it into a correlation matrix
- using small dedicated marginal generators instead of a generic hidden dispatcher
- applying theoretical standardization where the distributional moments are known analytically

## Future Vision

The long-term direction is a layered synthetic return generator in which:

- a scenario is declared through typed specifications
- each layer validates its own assumptions
- univariate marginals and cross-sectional dependence are composed explicitly
- regime or state logic can be added without changing lower layers
- each assumption can be tested in isolation

The intended outcome is a research-oriented toolbox that is easy to inspect, easy to extend, and honest about what each representation can and cannot express.
