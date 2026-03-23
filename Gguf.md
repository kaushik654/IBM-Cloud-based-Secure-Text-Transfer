
private fun copyToLocalStorage(filename: String): File {
    val storageFile = File(context.filesDir, filename)
    try {
        // Get actual asset size to compare
        val assetSize = context.assets.open(filename).use { it.available() }
        
        // Copy if file doesn't exist OR is incomplete (size mismatch)
        if (!storageFile.exists() || storageFile.length() != assetSize.toLong()) {
            storageFile.delete() // remove corrupted file first
            context.assets.open(filename).use { input ->
                storageFile.outputStream().use { output ->
                    input.copyTo(output, bufferSize = 8192)
                }
            }
            android.util.Log.d("SentenceEmbeddingProvider", "Successfully copied $filename")
        } else {
            android.util.Log.d("SentenceEmbeddingProvider", "Using cached $filename (${storageFile.length()} bytes)")
        }
    } catch (e: Exception) {
        android.util.Log.e("SentenceEmbeddingProvider", "Failed to copy $filename", e)
        storageFile.delete() // clean up on failure
        if (!storageFile.exists()) throw e
    }
    return storageFile
}
