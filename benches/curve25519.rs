use criterion::{criterion_group, criterion_main, Criterion, Fun};
use curve25519::{curve25519, curve25519_sk};

fn curve25519_bench_no_rand() {
    let random: [u8; 32] = [
        0x77, 0x07, 0x6a, 0x0a, 0x73, 0x18, 0xa5, 0x7d, 0x3c, 0x16, 0xc1, 0x72,
        0x51, 0xb2, 0x66, 0x45, 0xdf, 0x4c, 0x2f, 0x87, 0xeb, 0xc0, 0x99, 0x2a,
        0xb1, 0x77, 0xfb, 0xa5, 0x1d, 0xb9, 0x2c, 0x2a,
    ];
    let sk = curve25519_sk(Some(random)).unwrap();
    let mut basepoint: [u8; 32] = [0; 32];
    basepoint[0] = 9;
    let pk = basepoint;
    let _ = curve25519(sk, pk);
}

fn curve25519_bench_rand() {
    let sk = curve25519_sk(None).unwrap();
    let mut basepoint: [u8; 32] = [0; 32];
    basepoint[0] = 9;
    let pk = basepoint;
    let _ = curve25519(sk, pk);
}

fn criterion_benchmark(c: &mut Criterion) {
    let curve25519_no_rand = Fun::new("curve25519_bench_no_rand", |b, _| {
        b.iter(curve25519_bench_no_rand)
    });

    let curve25519_rand = Fun::new("curve25519_bench_rand", |b, _| {
        b.iter(curve25519_bench_rand)
    });

    c.bench_functions(
        "curve25519",
        vec![curve25519_no_rand, curve25519_rand],
        &0,
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
