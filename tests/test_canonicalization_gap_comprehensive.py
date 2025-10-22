#!/usr/bin/env python3
"""
Comprehensive canonicalization gap testing.

This exhaustively searches for gaps by:
1. Testing hundreds of random board configurations
2. Computing the FULL equivalence class for each
3. Comparing T→g exploration vs full exploration
4. Quantifying any missed states

Note: This is a slow test marked with @pytest.mark.slow
Use pytest -m "not slow" to skip in fast CI runs.
"""

import sys
from pathlib import Path
import numpy as np
from collections import defaultdict
import random
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from game.zertz_board import ZertzBoard


class EquivalenceClassAnalyzer:
    """Analyzes equivalence classes and canonicalization completeness."""

    def __init__(self, board):
        self.board = board
        self.canonicalizer = board.canonicalizer

    def compute_full_equivalence_class(self, state, max_iterations=1000):
        """
        Compute the full equivalence class using BFS.

        Explores all states reachable by ANY sequence of transformations.

        Args:
            state: Initial state
            max_iterations: Safety limit on BFS iterations

        Returns:
            dict: {state_bytes: path_description}
        """
        visited = {}  # state_bytes -> how we reached it
        queue = [(state, "identity")]
        iterations = 0

        symmetries = self.canonicalizer.get_all_symmetry_transforms()

        while queue and iterations < max_iterations:
            current_state, path = queue.pop(0)
            iterations += 1

            state_key = current_state.tobytes()

            if state_key in visited:
                continue

            visited[state_key] = path

            # Try all symmetries
            for sym_name, sym_fn in symmetries:
                if sym_name == "R0":
                    continue
                transformed = sym_fn(current_state)
                if transformed.tobytes() not in visited:
                    queue.append((transformed, f"{path}→{sym_name}"))

            # Try all valid translations
            translations = self.canonicalizer.get_all_translations(current_state)
            for trans_name, dy, dx in translations:
                if trans_name == "T0,0":
                    continue
                translated = self.canonicalizer.translate_state(current_state, dy, dx)
                if translated is not None and translated.tobytes() not in visited:
                    queue.append((translated, f"{path}→{trans_name}"))

        return visited

    def compute_T_then_g_class(self, state):
        """
        Compute equivalence class using only T→g (current approach).

        Returns:
            dict: {state_bytes: transform_name}
        """
        visited = {}
        symmetries = self.canonicalizer.get_all_symmetry_transforms()
        translations = self.canonicalizer.get_all_translations(state)

        # T→g: translate first, then symmetry
        for trans_name, dy, dx in translations:
            translated = self.canonicalizer.translate_state(state, dy, dx)
            if translated is None:
                continue

            for sym_name, sym_fn in symmetries:
                final = sym_fn(translated)
                transform = f"{trans_name}_{sym_name}" if trans_name != "T0,0" else sym_name
                visited[final.tobytes()] = transform

        return visited

    def compute_g_then_T_class(self, state):
        """
        Compute equivalence class using only g→T.

        Returns:
            dict: {state_bytes: transform_name}
        """
        visited = {}
        symmetries = self.canonicalizer.get_all_symmetry_transforms()

        # g→T: symmetry first, then translate
        for sym_name, sym_fn in symmetries:
            transformed = sym_fn(state)
            translations = self.canonicalizer.get_all_translations(transformed)

            for trans_name, dy, dx in translations:
                final = self.canonicalizer.translate_state(transformed, dy, dx)
                if final is None:
                    continue
                transform = f"{sym_name}_{trans_name}" if trans_name != "T0,0" else sym_name
                visited[final.tobytes()] = transform

        return visited

    def compute_both_orderings_class(self, state):
        """
        Compute equivalence class using BOTH T→g and g→T.

        Returns:
            dict: {state_bytes: transform_name}
        """
        T_then_g = self.compute_T_then_g_class(state)
        g_then_T = self.compute_g_then_T_class(state)

        # Merge (prefer simpler transform names)
        combined = T_then_g.copy()
        for state_key, transform in g_then_T.items():
            if state_key not in combined:
                combined[state_key] = transform

        return combined

    def analyze_coverage(self, state):
        """
        Analyze how well different approaches cover the equivalence class.

        Returns:
            dict with analysis results
        """
        # Compute different exploration strategies
        full = self.compute_full_equivalence_class(state)
        T_then_g = self.compute_T_then_g_class(state)
        g_then_T = self.compute_g_then_T_class(state)
        both = self.compute_both_orderings_class(state)

        # Convert to sets for comparison
        full_set = set(full.keys())
        T_then_g_set = set(T_then_g.keys())
        g_then_T_set = set(g_then_T.keys())
        both_set = set(both.keys())

        return {
            'full_size': len(full_set),
            'T_then_g_size': len(T_then_g_set),
            'g_then_T_size': len(g_then_T_set),
            'both_size': len(both_set),
            'T_then_g_coverage': len(T_then_g_set & full_set) / len(full_set) if full_set else 1.0,
            'g_then_T_coverage': len(g_then_T_set & full_set) / len(full_set) if full_set else 1.0,
            'both_coverage': len(both_set & full_set) / len(full_set) if full_set else 1.0,
            'T_then_g_only': len(T_then_g_set - g_then_T_set),
            'g_then_T_only': len(g_then_T_set - T_then_g_set),
            'both_only': len(both_set - full_set),
            'missed_by_T_then_g': len(full_set - T_then_g_set),
            'missed_by_g_then_T': len(full_set - g_then_T_set),
            'missed_by_both': len(full_set - both_set),
        }


def generate_random_board_state(rings=37, num_marbles=0, num_removed_rings=0, seed=None):
    """
    Generate a random board state.

    Args:
        rings: Board size
        num_marbles: Number of marbles to place
        num_removed_rings: Number of rings to remove
        seed: Random seed for reproducibility

    Returns:
        ZertzBoard with random state
    """
    rnd = random.Random(seed) if seed else random.Random()
    board = ZertzBoard(rings=rings)

    # Get valid positions
    valid_positions = []
    for y in range(board.width):
        for x in range(board.width):
            if board.state[0, y, x] == 1:  # Ring exists
                valid_positions.append((y, x))

    # Remove some rings
    if num_removed_rings > 0 and valid_positions:
        positions_to_remove = rnd.sample(valid_positions, min(num_removed_rings, len(valid_positions)))
        for y, x in positions_to_remove:
            board.state[0, y, x] = 0
            valid_positions.remove((y, x))

    # Place marbles
    if num_marbles > 0 and valid_positions:
        positions_for_marbles = rnd.sample(valid_positions, min(num_marbles, len(valid_positions)))
        for y, x in positions_for_marbles:
            marble_type = rnd.choice([1, 2, 3])  # white, grey, black
            board.state[marble_type, y, x] = 1

    return board


def examine_random_configurations(num_tests=10000, rings=37):
    """
    Test many random board configurations.

    Returns:
        dict with aggregate statistics
    """
    print(f"Testing {num_tests} random configurations...")
    print("=" * 80)

    analyzer_template = EquivalenceClassAnalyzer(ZertzBoard(rings=rings))

    results = {
        'total': 0,
        'gaps_found': 0,
        'T_then_g_perfect': 0,
        'both_perfect': 0,
        'coverage_T_then_g': [],
        'coverage_both': [],
        'max_gap_size': 0,
        'gap_examples': [],
    }

    rnd = random.Random()
    rnd.seed(6031769)
    for i in range(num_tests):
        # Generate random configuration
        num_marbles = rnd.randint(0, 8)
        num_removed = rnd.randint(0, 12)

        board = generate_random_board_state(
            rings=rings,
            num_marbles=num_marbles,
            num_removed_rings=num_removed,
            seed=i
        )

        analyzer = EquivalenceClassAnalyzer(board)
        analysis = analyzer.analyze_coverage(board.state)

        results['total'] += 1
        results['coverage_T_then_g'].append(analysis['T_then_g_coverage'])
        results['coverage_both'].append(analysis['both_coverage'])

        # Check for gaps
        if analysis['missed_by_T_then_g'] > 0:
            results['gaps_found'] += 1
            results['max_gap_size'] = max(results['max_gap_size'], analysis['missed_by_T_then_g'])

            if len(results['gap_examples']) < 5:
                results['gap_examples'].append({
                    'test_num': i,
                    'num_marbles': num_marbles,
                    'num_removed': num_removed,
                    'analysis': analysis
                })

        if analysis['T_then_g_coverage'] == 1.0:
            results['T_then_g_perfect'] += 1

        if analysis['both_coverage'] == 1.0:
            results['both_perfect'] += 1

        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_tests} tests...")

    return results

def print_summary(results):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    print(f"\nTests completed: {results['total']}")
    print(f"Gaps found: {results['gaps_found']} ({results['gaps_found']/results['total']*100:.1f}%)")
    print(f"Perfect coverage with T→g only: {results['T_then_g_perfect']} ({results['T_then_g_perfect']/results['total']*100:.1f}%)")
    print(f"Perfect coverage with both: {results['both_perfect']} ({results['both_perfect']/results['total']*100:.1f}%)")

    if results['coverage_T_then_g']:
        avg_coverage = np.mean(results['coverage_T_then_g'])
        min_coverage = np.min(results['coverage_T_then_g'])
        print(f"\nT→g coverage: avg={avg_coverage*100:.1f}%, min={min_coverage*100:.1f}%")

    if results['coverage_both']:
        avg_coverage = np.mean(results['coverage_both'])
        min_coverage = np.min(results['coverage_both'])
        print(f"Both orderings coverage: avg={avg_coverage*100:.1f}%, min={min_coverage*100:.1f}%")

    if results['gaps_found'] > 0:
        print(f"\n❌ GAPS CONFIRMED!")
        print(f"   Maximum gap size: {results['max_gap_size']} states")
        print(f"\n   Example gaps:")
        for ex in results['gap_examples'][:3]:
            print(f"   - Test #{ex['test_num']}: {ex['num_marbles']} marbles, {ex['num_removed']} removed")
            print(f"     Missed {ex['analysis']['missed_by_T_then_g']} states out of {ex['analysis']['full_size']}")
    else:
        print(f"\n✓ NO GAPS FOUND in {results['total']} tests")
        print(f"   Current T→g approach appears complete for tested configurations")


@pytest.mark.slow
def test_for_canonicalization_gaps():
    """Exhaustive canonicalization gap testing (slow).

    This test is marked as slow because it runs 100 random configurations
    with full BFS equivalence class exploration. Use pytest -m "not slow"
    to skip in regular CI runs.
    """
    print("COMPREHENSIVE CANONICALIZATION GAP ANALYSIS")
    print("=" * 80)
    print("\nThis will:")
    print("1. Test specific problematic cases")
    print("2. Test 100 random board configurations")
    print("3. Compute full equivalence classes via BFS")
    print("4. Compare T→g, g→T, and both approaches")
    print("5. Quantify any gaps found")
    print()

    # Test random configurations
    random_results = examine_random_configurations(num_tests=100, rings=37)

    # Print summary
    print_summary(random_results)

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if random_results['gaps_found'] > 0:
        print("\n❌ Gaps were found in the canonicalization!")
        print("   The current T→g approach misses some equivalent states.")
        print("\n   RECOMMENDATION: Implement both T→g AND g→T orderings")
        print("   - This ensures complete coverage of equivalence classes")
        print("   - Maximizes transposition table hits in MCTS")
        print("   - Eliminates possibility of missed transpositions")
        assert(False)
    else:
        print("\n✓ No gaps found in comprehensive testing")
        print("   The current T→g approach appears sufficient.")
        print("\n   RECOMMENDATION: Keep current implementation")
        print("   - Simpler code, faster canonicalization")
        print("   - Appears to cover full equivalence classes in practice")
        print("   - Can always add g→T later if needed")