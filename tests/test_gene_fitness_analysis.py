from app.optimize.gene_fitness_analysis import analyze_gene_fitness_correlation


class FakeGene:
    def __init__(self, label, is_int=False):
        self.label = label
        self.is_int = is_int


def test_not_enough_candidates_returns_note():
    genes = [FakeGene("period")]
    result = analyze_gene_fitness_correlation(genes, [([10], 1.0), ([20], 2.0)])
    assert result.note
    assert result.findings == []


def test_detects_positive_correlation():
    genes = [FakeGene("period", is_int=True)]
    # fitness increases directly with the gene's value
    population = [([v], float(v)) for v in range(1, 21)]
    result = analyze_gene_fitness_correlation(genes, population)
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.correlation > 0.9
    assert finding.direction == "higher values -> better fitness"
    assert finding.label == "period"


def test_detects_negative_correlation():
    genes = [FakeGene("stop_loss_pips")]
    population = [([v], -float(v)) for v in range(1, 21)]
    result = analyze_gene_fitness_correlation(genes, population)
    finding = result.findings[0]
    assert finding.correlation < -0.9
    assert finding.direction == "lower values -> better fitness"


def test_ignores_non_finite_fitness():
    genes = [FakeGene("period")]
    population = [([v], float(v)) for v in range(1, 21)] + [([999], float("-inf"))] * 5
    result = analyze_gene_fitness_correlation(genes, population)
    assert result.n_finite == 20
    assert len(result.findings) == 1


def test_no_relationship_is_labeled_clearly():
    genes = [FakeGene("noise_param")]
    # Alternating value with no relation to fitness pattern
    population = [([1 if i % 2 == 0 else 100], float(i % 3)) for i in range(20)]
    result = analyze_gene_fitness_correlation(genes, population)
    assert len(result.findings) == 1
    assert result.findings[0].direction in ("no clear relationship", "higher values -> better fitness", "lower values -> better fitness")


def test_summary_lines_skips_no_relationship_and_respects_top_n():
    genes = [FakeGene("a"), FakeGene("b"), FakeGene("c")]
    population = [([v, -v, 5], float(v)) for v in range(1, 21)]
    result = analyze_gene_fitness_correlation(genes, population)
    lines = result.summary_lines(top_n=1)
    assert len(lines) <= 1
    for line in lines:
        assert "no clear relationship" not in line


def test_summary_lines_with_note_returns_note_only():
    genes = [FakeGene("a")]
    result = analyze_gene_fitness_correlation(genes, [([1], 1.0)])
    assert result.summary_lines() == [result.note]
