import numpy as np
import pytest
from sklearn.datasets import make_classification
from uncertainty_quantification.main import AleatoricUncertainty, EpistemicUncertainty



def generate_test_data(n_samples=1000, n_features=5, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=3,
        random_state=random_state
    )
    return X, y


@pytest.fixture
def setup_data():
    return generate_test_data()


class TestAleatoricUncertainty:

    def test_initialization(self):
        # Test valid method
        au = AleatoricUncertainty(method='entropy')
        assert au.method == 'entropy'

        # Test invalid method
        with pytest.raises(ValueError):
            AleatoricUncertainty(method='invalid_method')

    def test_fit(self, setup_data):
        X, y = setup_data
        au = AleatoricUncertainty()
        au.fit(X, y)

        # Check if essential attributes are set
        assert hasattr(au, 'classifier')
        assert hasattr(au, 'n_classes')
        assert au.n_classes == 3

    def test_entropy_method(self, setup_data):
        X, y = setup_data
        au = AleatoricUncertainty(method='entropy')
        au.fit(X, y)
        probs, uncertainty = au.predict_uncertainty(X)

        # Check shapes
        assert probs.shape == (X.shape[0], 3)  # 3 classes
        assert uncertainty.shape == (X.shape[0],)

        # Check probability constraints
        assert np.allclose(np.sum(probs, axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)

        # Check uncertainty constraints
        assert np.all(uncertainty >= 0)
        max_entropy = np.log(3)  # Maximum entropy for 3 classes
        assert np.all(uncertainty <= max_entropy * 1.1)  # Adding 10% margin for numerical errors


class TestEpistemicUncertainty:

    def test_initialization(self):
        # Test valid method
        eu = EpistemicUncertainty(method='ensemble')
        assert eu.method == 'ensemble'

        # Test invalid method
        with pytest.raises(ValueError):
            EpistemicUncertainty(method='invalid_method')

    def test_ensemble_method(self, setup_data):
        """Test ensemble-based uncertainty quantification."""
        X, y = setup_data
        eu = EpistemicUncertainty(method='ensemble', n_estimators=3)
        eu.fit(X, y)
        probs, uncertainty = eu.predict_uncertainty(X)

        # Check shapes and constraints
        assert probs.shape == (X.shape[0], 3)
        assert uncertainty.shape == (X.shape[0],)
        assert np.all(uncertainty >= 0)
        assert np.all(probs >= 0) and np.all(probs <= 1)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

    def test_comparative_uncertainty(self, setup_data):
        """Test that both uncertainty types provide meaningful different information."""
        X, y = setup_data

        # Initialize both uncertainty types
        aleatoric = AleatoricUncertainty(method='entropy')
        epistemic = EpistemicUncertainty(method='ensemble')

        # Fit and predict
        aleatoric.fit(X, y)
        epistemic.fit(X, y)

        _, aleatoric_uncertainty = aleatoric.predict_uncertainty(X)
        _, epistemic_uncertainty = epistemic.predict_uncertainty(X)

        # Check correlation between uncertainties
        correlation = np.corrcoef(aleatoric_uncertainty, epistemic_uncertainty)[0, 1]

        # They should be somewhat correlated but not perfectly
        assert 0 < correlation < 0.95


if __name__ == '__main__':
    # Run some basic tests
    X, y = generate_test_data()

    print("Testing Aleatoric Uncertainty...")
    au = AleatoricUncertainty()
    au.fit(X, y)
    probs, uncertainty = au.predict_uncertainty(X)
    print(f"Aleatoric uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

    print("\nTesting Epistemic Uncertainty...")
    eu = EpistemicUncertainty()
    eu.fit(X, y)
    probs, uncertainty = eu.predict_uncertainty(X)
    print(f"Epistemic uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")