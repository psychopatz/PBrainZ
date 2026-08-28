from pbrainz.single_instance import SingleInstanceLock


def test_single_instance_lock_rejects_a_second_process_lock(tmp_path) -> None:
    path = tmp_path / "data" / ".p-brainz.lock"
    first = SingleInstanceLock(path)
    second = SingleInstanceLock(path)

    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()

    assert second.acquire() is True
    second.release()
