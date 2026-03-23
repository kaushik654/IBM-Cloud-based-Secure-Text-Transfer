private fun copyToLocalStorage(filename: String): File {
    val storageFile = File(parent = context.filesDir, child = filename)
    try {
        if (!storageFile.exists()) {
            context.assets.open(filename).use { input ->
                storageFile.outputStream().use { output ->
                    input.copyTo(output, bufferSize = 8192)
                }
            }
            android.util.Log.d("SentenceEmbeddingProvider", "Successfully copied $filename")
        } else {
            android.util.Log.d("SentenceEmbeddingProvider", "File already exists, skipping copy: $filename")
        }
    } catch (e: Exception) {
        android.util.Log.e("SentenceEmbeddingProvider", "Failed to copy $filename", e)
        if (!storageFile.exists()) throw e
    }
    return storageFile
}
