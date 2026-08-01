TINY = 1e-30


def abeles_torch(q, layers, scale=1.0, bkg=0.0, threads=1):
    # naive pytorch implementation of abeles calculation function
    import torch

    # might be being passed numpy arrays.
    if not torch.is_tensor(q):
        q = torch.asarray(q, dtype=torch.float64)
    qvals = q.to(torch.float64)

    if not torch.is_tensor(layers):
        layers = torch.asarray(layers, dtype=torch.float64)
    layers = layers.to(torch.float64)

    flatq = qvals.reshape(-1)

    nlayers = layers.shape[0] - 2
    npnts = flatq.numel()
    device = layers.device

    sld_vals = (
        (layers[1:, 1] - layers[0, 1]) + 1j * (torch.abs(layers[1:, 2]) + TINY)
    ) * 1.0e-6
    sld_vals = sld_vals.to(torch.complex128)
    sld = torch.cat(
        [torch.zeros(1, dtype=torch.complex128, device=device), sld_vals]
    )

    kn = torch.sqrt(
        flatq[None, :].to(torch.complex128) ** 2 / 4.0
        - 4.0 * torch.pi * sld[:, None]
    )
    kn_top = kn[:-1]
    kn_bot = kn[1:]

    d2 = layers[1:, 3][:, None] ** 2
    damping = torch.exp(-2.0 * kn_top * kn_bot * d2)
    rj = (kn_top - kn_bot) / (kn_top + kn_bot) * damping

    if nlayers:
        exponent = kn[1:-1] * 1j * torch.abs(layers[1:-1, 0])[:, None]
        mi00_inner = torch.exp(exponent)
        mi11_inner = torch.exp(-exponent)
        ones_row = torch.ones(
            (1, npnts), dtype=torch.complex128, device=device
        )
        mi00 = torch.cat([ones_row, mi00_inner], dim=0)
        mi11 = torch.cat([ones_row, mi11_inner], dim=0)
    else:
        mi00 = torch.ones((1, npnts), dtype=torch.complex128, device=device)
        mi11 = mi00

    mi10 = rj * mi00
    mi01 = rj * mi11

    mrtot00, mrtot01, mrtot10, mrtot11 = mi00[0], mi01[0], mi10[0], mi11[0]

    for i in range(nlayers):
        _mi00, _mi10, _mi01, _mi11 = (
            mi00[i + 1],
            mi10[i + 1],
            mi01[i + 1],
            mi11[i + 1],
        )
        p00 = mrtot00 * _mi00 + mrtot10 * _mi01
        p10 = mrtot00 * _mi10 + mrtot10 * _mi11
        p01 = mrtot01 * _mi00 + mrtot11 * _mi01
        p11 = mrtot01 * _mi10 + mrtot11 * _mi11
        mrtot00, mrtot01, mrtot10, mrtot11 = p00, p01, p10, p11

    r = mrtot01 / mrtot00
    reflectivity = r * torch.conj(r)
    return scale * torch.real(reflectivity).reshape(qvals.shape) + bkg
