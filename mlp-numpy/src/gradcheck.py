import numpy as np
from .losses import softmax_cross_entropy


def relative_error(g_num, g_ana):
    g_num, g_ana = np.asarray(g_num), np.asarray(g_ana)
    return np.linalg.norm(g_num - g_ana) / (
        np.linalg.norm(g_num) + np.linalg.norm(g_ana) + 1e-12
    )


def check_layer_gradients(model, X, y, eps=1e-5, max_params_per_tensor=50, seed=0):
    rng = np.random.default_rng(seed)

    loss, dZ = softmax_cross_entropy(model.forward(X), y)
    model.backward(dZ)

    results = []
    for li, layer in enumerate(model.layers):
        pg = layer.params_and_grads()
        names = ["W", "b"] if len(pg) == 2 else [f"p{k}" for k in range(len(pg))]

        for (param, grad), pname in zip(pg, names):
            k = min(max_params_per_tensor, param.size)
            idxs = rng.choice(param.size, size=k, replace=False)

            nums, anas = [], []
            for idx in idxs:
                old = param.flat[idx]

                param.flat[idx] = old + eps
                lp, _ = softmax_cross_entropy(model.forward(X), y)

                param.flat[idx] = old - eps
                lm, _ = softmax_cross_entropy(model.forward(X), y)

                param.flat[idx] = old          # trả về giá trị gốc

                nums.append((lp - lm) / (2 * eps))
                anas.append(grad.flat[idx])

            err = relative_error(nums, anas)
            results.append({"layer": li, "param": pname,
                            "rel_error": err, "passed": err < 1e-5})
    return results


def format_report(results):
    lines = ["| Tầng | Tham số | Sai số tương đối | Đạt (<1e-5) |",
             "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['layer']} | {r['param']} | {r['rel_error']:.2e} | "
                     f"{'✓' if r['passed'] else '✗'} |")
    return "\n".join(lines)
