private fun copyToLocalStorage(filename: String): File {
    val storageFile = File(context.filesDir, filename)
    try {
        context.assets.open(filename).use { input ->
            storageFile.outputStream().use { output ->
                input.copyTo(output, bufferSize = 8192) // streams in 8KB chunks
            }
        }
        android.util.Log.d("SentenceEmbeddingProvider", "Successfully copied $filename")
    } catch (e: Exception) {
        android.util.Log.e("SentenceEmbeddingProvider", "Failed to copy $filename", e)
        if (!storageFile.exists()) throw e
    }
    return storageFile
}
