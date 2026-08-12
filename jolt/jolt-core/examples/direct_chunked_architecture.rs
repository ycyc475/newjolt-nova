//! Minimal D1 runner: ELF -> streaming trace blocks -> direct prover seam.
//!
//! D1 is expected to stop with `UnsupportedRelation::Lookup`. Reaching that
//! error proves that no monolithic proving or verification API was invoked.

use std::{env, error::Error, fs, path::PathBuf};

use common::jolt_device::MemoryConfig;
use jolt_core::zkvm::block::{
    DirectChunkedConfig, DirectChunkedError, DirectChunkedPreprocessing, DirectChunkedProver,
    DirectRelation,
};

fn main() -> Result<(), Box<dyn Error>> {
    let elf_path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: direct_chunked_architecture <guest.elf> [block_capacity]")?;
    let block_capacity = env::args_os()
        .nth(2)
        .map(|value| value.to_string_lossy().parse::<usize>())
        .transpose()?
        .unwrap_or(1 << 12);
    let elf = fs::read(&elf_path)?;
    let memory_config = MemoryConfig::default();
    let blocks = tracer::trace_blocks(
        &elf,
        Some(&elf_path),
        &[],
        &[],
        &[],
        &memory_config,
        None,
        block_capacity,
    );
    let direct = DirectChunkedProver::new(
        DirectChunkedPreprocessing::from_program_bytes(&elf, usize::MAX),
        DirectChunkedConfig {
            block_capacity,
            compress_final_spartan: false,
        },
    )?;

    match direct.prove(blocks) {
        Err(DirectChunkedError::UnsupportedRelation {
            relation: DirectRelation::Lookup,
            audited_blocks,
        }) => {
            println!(
                "D1 architecture seam validated: streamed {audited_blocks} block(s); lookup is explicitly unsupported"
            );
            Ok(())
        }
        Err(other) => Err(other.into()),
        Ok(_) => Err("D1 unexpectedly emitted a production proof".into()),
    }
}
