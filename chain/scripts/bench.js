// Measures every on-chain number reported in the paper and writes them to
// artifacts/chain.json:  bond curve, burst detection, slashing split,
// per-function gas, and ERC-1155 batch amortisation.
const fs = require("fs");
const path = require("path");
const { ethers, network } = require("hardhat");

const OUT = path.join(__dirname, "..", "..", "artifacts");
const E = (v) => Number(ethers.formatEther(v));

async function gasOf(txp) {
  const r = await (await txp).wait();
  return Number(r.gasUsed);
}

async function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const signers = await ethers.getSigners();
  const [owner, pub, sybil, ...verifiers] = signers;
  const R = {};

  const F = await ethers.getContractFactory("ContentToken");
  const c = await F.deploy("ipfs://base/{id}.json");
  await c.waitForDeployment();
  R.deploy_gas = Number(
    (await ethers.provider.getTransactionReceipt(
      c.deploymentTransaction().hash
    )).gasUsed
  );

  // ---- 1. bond curve: bounded (deployed) vs unbounded (prior work) -------
  const reps = [0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000];
  const probe = signers[10];
  R.bond_curve = [];
  for (const p of reps) {
    await (await c.setReputation(probe.address, p)).wait();
    R.bond_curve.push({
      reputation: p,
      bounded: E(await c.bondOf(probe.address)),
      unbounded: E(await c.bondOfUnbounded(probe.address)),
    });
  }
  await (await c.setReputation(probe.address, 0)).wait();

  // risk-multiplier tiers on the bounded curve
  R.bond_by_tier = [];
  for (const p of [1, 10, 100, 1000]) {
    await (await c.setReputation(probe.address, p)).wait();
    const base = E(await c.bondOf(probe.address));
    R.bond_by_tier.push({ reputation: p, m1: base, m2: base * 2, m3: base * 3 });
  }

  // ---- 2. per-function gas ---------------------------------------------
  await (await c.setReputation(pub.address, 1000)).wait();
  const bond = await c.bondOf(pub.address);
  R.bond_pub_eth = E(bond);

  R.gas = {};
  R.gas.mintContent = await gasOf(
    c.connect(pub).mintContent("ipfs://a1", { value: bond })
  );
  R.gas.setReputation = await gasOf(c.setReputation(pub.address, 1000));

  // ---- 3. ERC-1155 batch amortisation ----------------------------------
  R.batch = [];
  for (const n of [1, 2, 4, 8, 16, 32]) {
    const uris = Array.from({ length: n }, (_, i) => `ipfs://b${n}-${i}`);
    const g = await gasOf(
      c.connect(pub).mintBatch(uris, { value: bond * BigInt(n) })
    );
    R.batch.push({ n, total: g, per_item: g / n });
  }
  const seq = R.batch[0].per_item;
  R.batch = R.batch.map((b) => ({
    ...b,
    reduction_pct: (100 * (seq - b.per_item)) / seq,
  }));

  // ---- 4. Sybil burst detection ----------------------------------------
  await (await c.setReputation(sybil.address, 10)).wait();
  const sBond = await c.bondOf(sybil.address);
  R.sybil = { bond_before: E(sBond), events: [] };
  for (let i = 1; i <= 6; i++) {
    let status;
    try {
      const need = await c.bondOf(sybil.address);
      await (
        await c.connect(sybil).mintContent(`ipfs://s${i}`, { value: sBond })
      ).wait();
      status = "accepted";
      R.sybil.events.push({
        mint: i,
        required_eth: E(need),
        sent_eth: E(sBond),
        status,
      });
    } catch (e) {
      const need = await c.bondOf(sybil.address);
      R.sybil.events.push({
        mint: i,
        required_eth: E(need),
        sent_eth: E(sBond),
        status: "reverted: bond too low",
      });
      break;
    }
  }
  R.sybil.riskMult_after = Number(await c.riskMult(sybil.address));
  R.sybil.bond_after = E(await c.bondOf(sybil.address));
  const pend = Number(await c.probationEnd(sybil.address));
  const now = (await ethers.provider.getBlock("latest")).timestamp;
  R.sybil.probation_seconds = pend > 0 ? pend - now : 0;

  // cost-to-corrupt: honest publisher vs Sybil fleet of k identities
  R.cost_to_corrupt = [];
  for (const k of [1, 5, 10, 25, 50, 100]) {
    await (await c.setReputation(probe.address, 0)).wait();
    const fresh = E(await c.bondOf(probe.address));
    R.cost_to_corrupt.push({
      identities: k,
      normal_eth: k * fresh,
      probation_eth: k * fresh * 3,
    });
  }

  // ---- 5. slashing distribution ----------------------------------------
  await (await c.setReputation(pub.address, 1000)).wait();
  const b2 = await c.bondOf(pub.address);
  const rc = await (
    await c.connect(pub).mintContent("ipfs://slashme", { value: b2 })
  ).wait();
  const id = Number(await c.nextId());
  const vs = verifiers.slice(0, 3).map((v) => v.address);
  const before = await Promise.all(
    vs.map((a) => ethers.provider.getBalance(a))
  );
  R.gas.slashStake = await gasOf(c.slashStake(id, vs));
  const after = await Promise.all(vs.map((a) => ethers.provider.getBalance(a)));
  R.slashing = {
    pot_eth: E(b2),
    recipients: vs.map((a, i) => ({
      verifier: `Verifier ${i + 1}`,
      received_eth: E(after[i] - before[i]),
      share_pct: (100 * Number(after[i] - before[i])) / Number(b2),
    })),
    publisher_reputation_after: Number(await c.reputation(pub.address)),
  };

  R.gas_price_gwei = 50;
  R.eth_usd = 3000;
  fs.writeFileSync(path.join(OUT, "chain.json"), JSON.stringify(R, null, 2));
  console.log(JSON.stringify(R, null, 2));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
