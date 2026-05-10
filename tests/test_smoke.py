"""Smoke tests — verify neurogolf_2026 imports."""


def test_import_package():
    import neurogolf_2026 as pkg

    assert pkg.__version__
