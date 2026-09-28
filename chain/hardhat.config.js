require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: {
    version: "0.8.20",
    settings: { optimizer: { enabled: true, runs: 200 } },
  },
  networks: {
    hardhat: {
      accounts: { count: 20, accountsBalance: "10000000000000000000000000" },
      allowUnlimitedContractSize: true,
    },
  },
};
