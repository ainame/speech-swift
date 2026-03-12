import Foundation
import HuggingFace
import os

/// Download errors
public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        }
    }
}

/// HuggingFace model downloader — shared between ASR, TTS, VAD, etc.
///
/// Uses `HubClient` from swift-huggingface for downloads while keeping
/// speech-swift's existing cache-directory layout stable.
public enum HuggingFaceDownloader {

    // MARK: - Cache Directory

    /// Get cache directory for a model.
    ///
    /// Returns the old flat cache path if it already contains model files (preserving
    /// ~10 GB of existing cached models), otherwise returns the namespaced cache path.
    public static func getCacheDirectory(for modelId: String, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = resolveBaseCacheDir(cacheDirName: cacheDirName)
        let fm = FileManager.default

        // Check old (flat) cache path for backward compat:
        //   ~/Library/Caches/qwen3-speech/aufklarer_Qwen3-ASR-0.6B-MLX-4bit/
        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if weightsExist(in: oldDir) {
            return oldDir
        }

        let dir = modelCacheDirectory(for: modelId, base: base)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Weight Existence Check

    /// Check if safetensors weights exist in a directory.
    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            AudioLog.download.debug("Could not list directory \(directory.path): \(error)")
            contents = []
        }
        return contents.contains { $0.pathExtension == "safetensors" }
    }

    // MARK: - Download

    /// Download model files from HuggingFace using `HubClient.downloadSnapshot(...)`.
    ///
    /// Builds glob patterns from the file list:
    /// - Always includes `config.json`
    /// - If `additionalFiles` doesn't contain `.safetensors` files, adds `*.safetensors`
    ///   and `model.safetensors.index.json` to discover sharded weights automatically
    /// - All entries in `additionalFiles` are added as-is (they work as glob patterns)
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        var globs: [String] = ["config.json"]

        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            globs.append("*.safetensors")
            globs.append("model.safetensors.index.json")
        }
        for file in additionalFiles where !globs.contains(file) {
            globs.append(file)
        }

        let client = makeHubClient(for: directory)
        guard let repo = Repo.ID(rawValue: modelId) else {
            throw DownloadError.failedToDownload("\(modelId): invalid repository identifier")
        }

        do {
            _ = try await client.downloadSnapshot(
                of: repo,
                to: directory,
                matching: globs
            ) { progress in
                progressHandler?(progress.fractionCompleted)
            }
        } catch {
            throw DownloadError.failedToDownload("\(modelId): \(error.localizedDescription)")
        }
    }

    // MARK: - Security Helpers (kept for backward compat + security tests)

    /// Convert an arbitrary modelId into a single, safe path component for on-disk caching.
    public static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    /// Validate that a remote file name is safe.
    public static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    /// Validate that a local path stays within the expected directory.
    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    // MARK: - Private Helpers

    /// Resolve the base cache directory from env vars or system default.
    private static func resolveBaseCacheDir(cacheDirName: String) -> URL {
        let fm = FileManager.default
        let root: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Legacy env var support
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }
        return root.appendingPathComponent(cacheDirName, isDirectory: true)
    }

    private static func modelCacheDirectory(for modelId: String, base: URL) -> URL {
        guard let repo = Repo.ID(rawValue: modelId) else {
            return base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        }

        return base
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(repo.namespace, isDirectory: true)
            .appendingPathComponent(repo.name, isDirectory: true)
    }

    private static func makeHubClient(for repoDir: URL) -> HubClient {
        let cacheRoot = resolveHubCacheRoot(for: repoDir)
        return HubClient(cache: HubCache(cacheDirectory: cacheRoot))
    }

    private static func resolveHubCacheRoot(for repoDir: URL) -> URL {
        let standardizedRepoDir = repoDir.standardizedFileURL
        let parent = standardizedRepoDir.deletingLastPathComponent()
        if parent.lastPathComponent != "models" {
            return parent.appendingPathComponent("hub", isDirectory: true)
        }

        return parent.deletingLastPathComponent().appendingPathComponent("hub", isDirectory: true)
    }
}
