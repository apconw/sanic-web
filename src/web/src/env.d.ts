/// <reference types="vite/client" />

interface ImportMetaEnv extends Readonly<Record<string, string>> {
  readonly VITE_BASE_API: string
  readonly VITE_SPARK_KEY: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
