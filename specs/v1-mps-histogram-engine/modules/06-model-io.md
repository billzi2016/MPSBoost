# Module Design: Model Format

## 1. Goal
Provide a stable, verifiable model format decoupled from training sessions so that
predictions agree before and after saving and CPU and MPS inference share one model.

## 2. File Structure
Version 0.2.0 uses an explicitly versioned binary container or JSON metadata plus
binary arrays and must include:
- magic;
- major/minor format version;
- endian and scalar type;
- objective, base-score, and learning-rate semantics;
- feature count and bin boundaries;
- tree count and each tree's node interval;
- flat node arrays and leaf values;
- segment lengths, offsets, and integrity validation.

Models must not contain raw training data, labels, absolute paths, unique device
identifiers, or cache contents.

## 3. Compatibility Strategy
- The same major version permits defined forward-compatible fields;
- reject unknown majors by default;
- loaders do not guess missing fields;
- conversion layers handle runtime internal-layout changes; files must not depend on
  GPU workspace layout;
- retain historical model fixtures in post-release tests.

## 4. Safe Loading
Check total length before parsing; check every count, offset, and multiplication for
overflow; validate every node child index, feature index, and boundary interval.
Loading failures do not modify existing model state.

## 5. Atomic Saving
Write a temporary file in the same directory; after flush and validation, atomically
replace the target. On failure retain the original file and clean project temporary
files. Do not assume rename atomicity across file systems.

## 6. Acceptance
- Saved and loaded predictions agree;
- reject truncation, random bytes, oversized lengths, and invalid indices;
- old-version compatibility fixtures pass;
- files do not leak training data;
- CPU and MPS read the same model.
