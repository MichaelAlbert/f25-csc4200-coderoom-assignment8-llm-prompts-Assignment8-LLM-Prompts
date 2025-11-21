#!/usr/bin/env python3
"""
delta_sync.py — model-delta creation, broadcast and merge helpers
Used by prompt_node.py with udp_overlay.py
"""

import os, io, torch, hashlib, tempfile, time
from udp_overlay import PeerNode, BROADCAST_IP, PORT
from update_exchanges import announce_model_meta, fragment_and_send, handle_incoming_chunk, receive_and_reassemble


# ---------- create and send deltas ----------
def export_delta(model, threshold=1e-6):
    """Compute sparse delta from current weights vs base.pt."""
    base = torch.load("base.pt", map_location="cpu")
    now = model.state_dict()
    delta = {}
    for k, v in now.items():
        diff = (v - base[k])
        if torch.norm(diff) > threshold:
            delta[k] = diff.half()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    torch.save(delta, tmp.name)
    sha = hashlib.sha256(open(tmp.name, "rb").read()).hexdigest()
    size = os.path.getsize(tmp.name)
    return tmp.name, sha, size



def broadcast_delta(node, path, sha, size):
    """
    broadcast your message delta
    """
    ver = f"v{int(time.time())}"
    chunks = (size + 1479) // 1480
    node._announce_model_meta(ver, size, chunks, sha)
    node._model_buffers[ver] = {"total": chunks, "parts": {}, "sha256": sha}
    fragment_and_send(node, ver, path, (BROADCAST_IP, PORT))
    print(f"[SEND] delta {ver} ({size} B)broadcasted")
    os.remove(path)
    


def reassemble_delta(node: PeerNode, ver: str):
    """
    reassemble your delta before you pass it to apply incoming delta
    """
    buf = node._model_buffers.get(ver)
    if buf is None:
        return None

    total = buf["total"]
    parts = buf["parts"]
    expected_sha = buf.get("sha256")

    if len(parts) < total:
        return None
    try:
        ordered = [parts[i] for i in range(total)]
        data = b"".join(ordered)
    except Exception as e:
        print(f"[ERROR] failed to reassemble {ver}: {e}")
        return None

    actual_sha = hashlib.sha256(data).hexdigest()
    if expected_sha != actual_sha:
        print(f"[WARN] SHA mismatch for {ver}: expected {expected_sha}, got {actual_sha}")
        return None
    return data






def apply_incoming_deltas(node, model, merge_weight=1.0):
    merged = 0

    for ver, buf in list(node._model_buffers.items()):
        last_idx  = buf.get("_last_idx", 0)
        last_total = buf["total"]


        data = reassemble_delta(node, ver)
        if data is None:
            continue


        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
        tmp.write(data)
        tmp.close()

        try:
            delta = torch.load(tmp.name, map_location="cpu", weights_only=False)
            sd = model.state_dict()
            applied = 0

            for k, v in delta.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = sd[k] + (merge_weight * v.to(sd[k].dtype))
                    applied += 1

            model.load_state_dict(sd)
            torch.save(model.state_dict(), "base.pt")
            print(f"[MERGE] model {ver} applied ✓ ({applied} layers)")
            merged += 1

        except Exception as e:
            print(f"[ERROR] failed to merge {ver}: {e}")

        finally:
            os.remove(tmp.name)
            node._model_buffers.pop(ver, None)

    return merged

