//! Validated D8 comparison artifact for native Jolt, Stage 19, and the direct
//! single-data-flow prover.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

use super::{
    stage19_hex_digest, DirectProofSizeBreakdown, DirectProvingMetrics, Stage19BenchmarkSample,
};

pub const DIRECT_D8_BENCHMARK_SCHEMA_VERSION: &str = "direct-chunked-jolt-nova-d8-v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectD8BaselineMeasurement {
    pub flow: String,
    pub proof_bytes: usize,
    pub prove_micros: u64,
    pub verify_micros: u64,
    pub peak_rss_delta_bytes: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectD8Measurement {
    pub proof_verified: bool,
    pub proof_size: DirectProofSizeBreakdown,
    pub prove_micros: u64,
    pub verify_micros: u64,
    pub peak_rss_delta_bytes: Option<u64>,
    pub metrics: DirectProvingMetrics,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirectD8BenchmarkArtifact {
    pub schema_version: String,
    pub workload: String,
    pub workload_sha3_256: String,
    pub block_capacity: usize,
    pub native_jolt: DirectD8BaselineMeasurement,
    pub stage19_dual_flow: DirectD8BaselineMeasurement,
    pub direct: DirectD8Measurement,
    pub artifact_digest: String,
}

impl DirectD8BenchmarkArtifact {
    pub fn from_stage19_sample(
        sample: &Stage19BenchmarkSample,
        direct: DirectD8Measurement,
    ) -> Result<Self, String> {
        sample.validate()?;
        let mut artifact = Self {
            schema_version: DIRECT_D8_BENCHMARK_SCHEMA_VERSION.to_string(),
            workload: sample.workload.clone(),
            workload_sha3_256: sample.workload_sha3_256.clone(),
            block_capacity: sample.block_target_size,
            native_jolt: DirectD8BaselineMeasurement {
                flow: "native-jolt".to_string(),
                proof_bytes: sample.proof_sizes.native_jolt_proof,
                prove_micros: sample.timings_micros.native_jolt_prove,
                verify_micros: sample.timings_micros.native_jolt_verify_and_export,
                peak_rss_delta_bytes: sample.memory_bytes.native_jolt_peak_delta,
            },
            stage19_dual_flow: DirectD8BaselineMeasurement {
                flow: "stage19-dual-data-flow".to_string(),
                proof_bytes: sample.proof_sizes.recursive_proof_payload,
                prove_micros: sample.timings_micros.recursive_extension(),
                verify_micros: sample.timings_micros.final_end_to_end_verify,
                peak_rss_delta_bytes: sample.memory_bytes.recursive_peak_delta,
            },
            direct,
            artifact_digest: String::new(),
        };
        artifact.artifact_digest = artifact.compute_digest()?;
        artifact.validate()?;
        Ok(artifact)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != DIRECT_D8_BENCHMARK_SCHEMA_VERSION {
            return Err("D8 benchmark schema version mismatch".to_string());
        }
        if self.workload.is_empty() || self.workload_sha3_256.len() != 64 {
            return Err("D8 benchmark workload identity is incomplete".to_string());
        }
        if self.block_capacity < 2 || !self.block_capacity.is_power_of_two() {
            return Err("D8 block capacity must be a power of two of at least two".to_string());
        }
        for baseline in [&self.native_jolt, &self.stage19_dual_flow] {
            if baseline.proof_bytes == 0
                || baseline.prove_micros == 0
                || baseline.verify_micros == 0
            {
                return Err(format!("{} baseline is incomplete", baseline.flow));
            }
        }
        if !self.direct.proof_verified
            || self.direct.proof_size.total_bytes == 0
            || self.direct.prove_micros == 0
            || self.direct.verify_micros == 0
        {
            return Err("D8 direct measurement is incomplete or unverified".to_string());
        }
        let metrics = &self.direct.metrics;
        if metrics.source_block_count == 0
            || metrics.block_count == 0
            || metrics.total_active_cycles == 0
            || !metrics.trace_residency_is_bounded(self.block_capacity)
            || metrics.spartan_proof_bytes == 0
            || metrics.retained_relation_subclaims != metrics.block_count.saturating_mul(4)
        {
            return Err("D8 direct structural metrics do not prove bounded execution".to_string());
        }
        if self.artifact_digest != self.compute_digest()? {
            return Err("D8 benchmark artifact digest mismatch".to_string());
        }
        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String, String> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn to_markdown(&self) -> Result<String, String> {
        self.validate()?;
        let rss = |value: Option<u64>| {
            value
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "n/a".to_string())
        };
        Ok(format!(
            "# Direct Chunked Jolt-Nova D8\n\nWorkload: `{}`  \nBlock capacity: `{}`  \nArtifact digest: `{}`\n\n| Flow | Proof bytes | Prove us | Verify us | Peak RSS delta bytes |\n|---|---:|---:|---:|---:|\n| Native Jolt | {} | {} | {} | {} |\n| Stage 19 dual flow | {} | {} | {} | {} |\n| Direct chunked | {} | {} | {} | {} |\n\n## Direct memory decomposition\n\n- source trace blocks: {}\n- fixed-capacity proof blocks: {}\n- trace passes: {}\n- max source trace cycles: {}\n- max resident trace blocks: {}\n- max resident trace cycles: {}\n- estimated peak resident trace bytes: {}\n- spooled trace bytes: {}\n- tracked RAM addresses: {}\n- initial RAM registry bytes: {}\n- retained relation subclaims: {}\n- PCS polynomial bytes: {}\n- RSS after capture: {}\n- RSS after relations: {}\n- RSS after PCS: {}\n- peak observed RSS: {}\n",
            self.workload,
            self.block_capacity,
            self.artifact_digest,
            self.native_jolt.proof_bytes,
            self.native_jolt.prove_micros,
            self.native_jolt.verify_micros,
            rss(self.native_jolt.peak_rss_delta_bytes),
            self.stage19_dual_flow.proof_bytes,
            self.stage19_dual_flow.prove_micros,
            self.stage19_dual_flow.verify_micros,
            rss(self.stage19_dual_flow.peak_rss_delta_bytes),
            self.direct.proof_size.total_bytes,
            self.direct.prove_micros,
            self.direct.verify_micros,
            rss(self.direct.peak_rss_delta_bytes),
            self.direct.metrics.source_block_count,
            self.direct.metrics.block_count,
            self.direct.metrics.trace_passes,
            self.direct.metrics.max_source_trace_cycles,
            self.direct.metrics.max_resident_trace_blocks,
            self.direct.metrics.max_resident_trace_cycles,
            self.direct.metrics.estimated_peak_resident_trace_bytes,
            self.direct.metrics.spooled_trace_bytes,
            self.direct.metrics.peak_tracked_ram_addresses,
            self.direct.metrics.initial_ram_registry_bytes,
            self.direct.metrics.retained_relation_subclaims,
            self.direct.metrics.pcs_polynomial_bytes,
            self.direct
                .metrics
                .after_capture_physical_memory_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.direct
                .metrics
                .after_relations_physical_memory_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.direct
                .metrics
                .after_pcs_physical_memory_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
            self.direct
                .metrics
                .peak_observed_physical_memory_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "n/a".to_string()),
        ))
    }

    fn compute_digest(&self) -> Result<String, String> {
        let encoded = serde_json::to_vec(&(
            &self.schema_version,
            &self.workload,
            &self.workload_sha3_256,
            self.block_capacity,
            &self.native_jolt,
            &self.stage19_dual_flow,
            &self.direct,
        ))
        .map_err(|error| error.to_string())?;
        Ok(stage19_hex_digest(Sha3_256::digest(encoded).into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::block::{
        Stage19MemoryBytes, Stage19ProofSizes, Stage19RelationMeasurement, Stage19TimingsMicros,
        JOLT_NOVA_STAGE19_LOOKUP_BACKEND, JOLT_NOVA_STAGE19_PROFILE_METHOD,
    };

    fn stage19_sample() -> Stage19BenchmarkSample {
        let relations = [
            "cpu-r1cs",
            "trace-io-claims",
            "register-read-write",
            "ram-read-write",
            "jolt-lasso-lookup-claim",
            "verified-jolt-receipt-binding",
            "nova-block-fold",
            "spartan-final-compression",
        ];
        Stage19BenchmarkSample {
            run_index: 1,
            workload: "fibonacci-2".to_string(),
            workload_sha3_256: "11".repeat(32),
            block_target_size: 2,
            block_count: 1,
            active_cycles: 1,
            padded_trace_length: 2,
            max_resident_trace_blocks: 1,
            max_resident_trace_cycles: 1,
            receipt_is_production_zk: true,
            native_jolt_verified: true,
            streaming_proof_verified: true,
            complete_zk_verified: true,
            lookup_backend: JOLT_NOVA_STAGE19_LOOKUP_BACKEND.to_string(),
            receipt_digest: "22".repeat(32),
            jolt_statement_id: "33".repeat(32),
            timings_micros: Stage19TimingsMicros {
                trace: 1,
                native_jolt_prove: 10,
                native_jolt_verify_and_export: 5,
                nova_spartan_stream: 8,
                recursive_blindfold_prove: 7,
                final_end_to_end_verify: 4,
                measured_total: 35,
            },
            throughput_active_cycles_per_second: 1_000_000.0 / 35.0,
            memory_bytes: Stage19MemoryBytes {
                native_jolt_start: Some(1),
                native_jolt_peak: Some(3),
                native_jolt_peak_delta: Some(2),
                recursive_start: Some(3),
                recursive_peak: Some(7),
                recursive_peak_delta: Some(4),
                estimated_peak_trace: 128,
                peak_tracked_ram_addresses: 1,
            },
            proof_sizes: Stage19ProofSizes {
                native_jolt_proof: 10,
                folded_spartan_proof: 5,
                blindfold_spartan_proof: 6,
                recursive_proof_payload: 11,
            },
            relation_profiles: relations
                .into_iter()
                .map(|relation| Stage19RelationMeasurement {
                    relation: relation.to_string(),
                    attribution_method: JOLT_NOVA_STAGE19_PROFILE_METHOD.to_string(),
                    calls: 1,
                    total_nanos: 1,
                    max_nanos: 1,
                })
                .collect(),
        }
    }

    fn direct_measurement() -> DirectD8Measurement {
        DirectD8Measurement {
            proof_verified: true,
            proof_size: DirectProofSizeBreakdown {
                spartan_compressed_bytes: 5,
                total_bytes: 20,
                ..DirectProofSizeBreakdown::default()
            },
            prove_micros: 9,
            verify_micros: 3,
            peak_rss_delta_bytes: Some(8),
            metrics: DirectProvingMetrics {
                source_block_count: 1,
                block_count: 1,
                total_active_cycles: 1,
                trace_passes: 2,
                max_resident_trace_blocks: 1,
                max_resident_trace_cycles: 1,
                max_source_trace_cycles: 1,
                retained_relation_subclaims: 4,
                spartan_proof_bytes: 5,
                ..DirectProvingMetrics::default()
            },
        }
    }

    #[test]
    fn d8_comparison_artifact_validates_and_detects_tampering() {
        let artifact =
            DirectD8BenchmarkArtifact::from_stage19_sample(&stage19_sample(), direct_measurement())
                .unwrap();
        artifact.validate().unwrap();
        assert!(artifact.to_markdown().unwrap().contains("Direct chunked"));

        let mut tampered = artifact;
        tampered.direct.metrics.max_resident_trace_blocks = 3;
        assert!(tampered.validate().is_err());
    }
}
