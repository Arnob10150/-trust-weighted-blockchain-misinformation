// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC1155/ERC1155.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/// @title Trust-weighted content provenance with a bounded integrity bond.
/// @notice One ERC-1155 id per published item. Publication escrows a bond whose
///         size falls with publisher reputation and rises with assessed risk.
contract ContentToken is ERC1155, ReentrancyGuard, Ownable {
    uint256 public constant BASE_BOND = 1000 ether;  // B_base
    uint256 public constant SCALE = 10000;           // fixed-point scale
    uint256 public constant EPS = 1;                 // epsilon, avoids /0

    // Bounded-curve parameters (Section III-B). RATIO_CAP bounds the
    // novice/elite bond ratio so authentic newcomers are not priced out.
    uint256 public constant RATIO_CAP = 20;
    uint256 public constant BOND_FLOOR = BASE_BOND / RATIO_CAP;

    uint256 public constant BURST_WINDOW = 1 hours;
    uint256 public constant BURST_LIMIT = 5;
    uint256 public constant PROBATION = 30 days;
    uint8 public constant PROBATION_MULT = 3;

    uint256 public nextId;

    mapping(address => uint256) public reputation;    // P_user, TDSW-derived
    mapping(address => uint8) public riskMult;        // M_risk
    mapping(address => uint256) public probationEnd;
    mapping(address => uint256[]) private mintTimes;

    mapping(uint256 => uint256) public escrow;
    mapping(uint256 => address) public publisherOf;
    mapping(uint256 => bool) public settled;

    event Published(uint256 indexed id, address indexed pub, uint256 bond);
    event ProbationTriggered(address indexed pub, uint256 until);
    event Slashed(uint256 indexed id, uint256 pot, uint256 perVerifier);

    constructor(string memory uri_) ERC1155(uri_) Ownable() {}

    /// @notice Log-bounded inverse bonding curve, Eq. (5).
    ///         B = max(BOND_FLOOR, B_base * M / (1 + log2(1 + P)))
    function bondOf(address u) public view returns (uint256) {
        uint8 m = riskMult[u] == 0 ? 1 : riskMult[u];
        uint256 denom = SCALE + _log2Fixed(reputation[u] + EPS);
        uint256 b = (BASE_BOND * m * SCALE) / denom;
        uint256 floor_ = BOND_FLOOR * m;
        return b < floor_ ? floor_ : b;
    }

    /// @notice Unbounded hyperbolic curve of prior work, Eq. (2). Retained for
    ///         comparison only: its novice/elite ratio is unbounded.
    function bondOfUnbounded(address u) public view returns (uint256) {
        uint8 m = riskMult[u] == 0 ? 1 : riskMult[u];
        return (BASE_BOND * m) / (reputation[u] + EPS);
    }

    function setReputation(address u, uint256 p) external onlyOwner {
        reputation[u] = p;
    }

    function mintContent(string calldata uri_)
        external
        payable
        nonReentrant
        returns (uint256)
    {
        // The bond is priced with the multiplier in force *before* this call,
        // and the burst check is applied afterwards. Pricing first and
        // detecting second is deliberate: if the raised bond were enforced on
        // the very call that trips the detector, the revert would roll back
        // the probation write and the flag would never persist. The multiplier
        // therefore binds prospectively, from the next publication onward.
        uint256 bond = bondOf(msg.sender);
        require(msg.value >= bond, "bond too low");
        _detectBurst(msg.sender);
        uint256 id = ++nextId;
        escrow[id] = bond;
        publisherOf[id] = msg.sender;
        _mint(msg.sender, id, 1, bytes(uri_));
        if (msg.value > bond) {
            (bool ok, ) = msg.sender.call{value: msg.value - bond}("");
            require(ok, "refund failed");
        }
        emit Published(id, msg.sender, bond);
        return id;
    }

    /// @notice Batch publication: amortises the ERC-1155 write path.
    function mintBatch(string[] calldata uris)
        external
        payable
        nonReentrant
        returns (uint256[] memory)
    {
        uint256 n = uris.length;
        uint256 bond = bondOf(msg.sender);
        require(msg.value >= bond * n, "bond too low");
        uint256[] memory ids = new uint256[](n);
        uint256[] memory amts = new uint256[](n);
        for (uint256 i = 0; i < n; i++) {
            uint256 id = ++nextId;
            ids[i] = id;
            amts[i] = 1;
            escrow[id] = bond;
            publisherOf[id] = msg.sender;
        }
        _mintBatch(msg.sender, ids, amts, "");
        return ids;
    }

    function _detectBurst(address u) internal {
        uint256[] storage t = mintTimes[u];
        t.push(block.timestamp);
        uint256 n = t.length;
        if (n >= BURST_LIMIT && t[n - 1] - t[n - BURST_LIMIT] < BURST_WINDOW) {
            riskMult[u] = PROBATION_MULT;
            probationEnd[u] = block.timestamp + PROBATION;
            emit ProbationTriggered(u, probationEnd[u]);
        }
    }

    /// @notice Seize the escrowed bond of a disproven item and split it evenly
    ///         among the verifiers who carried the challenge.
    function slashStake(uint256 id, address[] calldata verifiers)
        external
        onlyOwner
        nonReentrant
    {
        require(!settled[id], "settled");
        require(verifiers.length > 0, "no verifiers");
        settled[id] = true;
        uint256 pot = escrow[id];
        escrow[id] = 0;
        address pub = publisherOf[id];
        reputation[pub] = (reputation[pub] * 80) / 100;   // -20% reputation
        uint256 share = pot / verifiers.length;
        for (uint256 i = 0; i < verifiers.length; i++) {
            (bool ok, ) = payable(verifiers[i]).call{value: share}("");
            require(ok, "payout failed");
        }
        emit Slashed(id, pot, share);
    }

    /// @dev Integer log2 scaled by SCALE, monotone and cheap on the EVM.
    function _log2Fixed(uint256 x) internal pure returns (uint256) {
        uint256 n = 0;
        while (x > 1) {
            x >>= 1;
            n++;
        }
        return n * SCALE;
    }

    receive() external payable {}
}
