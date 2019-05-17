# Curve25519 in Pure Rust

[![Build Status](https://travis-ci.org/shekohex/curve25519-rs.svg?branch=master)](https://travis-ci.org/shekohex/curve25519-rs) [![Documentation](https://img.shields.io/badge/docs-0.1.0-blue.svg)](https://shadykhalifa.me/curve25519-rs/) [![License](https://img.shields.io/badge/license-MIT%2FApache--2-yellowgreen.svg)](#)
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fshekohex%2Fcurve25519-rs.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fshekohex%2Fcurve25519-rs?ref=badge_shield)


Curve25519 is a state-of-the-art Diffie-Hellman function suitable for a wide variety of applications.

Given a user's 32-byte secret key, Curve25519 computes the user's 32-byte public key.
Given the user's 32-byte secret key and another user's 32-byte public key, Curve25519 computes a 32-byte secret shared by the two users.
This secret can then be used to authenticate and encrypt messages between the two users.

See [spec](https://cr.yp.to/ecdh.html)

## Usage
in your `Cargo.toml` file

```toml
[dependencies]
curve25519 = { git = "https://github.com/shekohex/curve25519-rs" }
```

### Some Notes
This crate was extracted from [rust-crypto](https://github.com/DaGenix/rust-crypto) crate.

## License

All crates licensed under either of

 * [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
 * [MIT license](http://opensource.org/licenses/MIT)


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fshekohex%2Fcurve25519-rs.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fshekohex%2Fcurve25519-rs?ref=badge_large)